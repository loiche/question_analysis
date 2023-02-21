import re
import os
import sys
import MeCab
import random
import shutil
import urllib
import plotly
import datetime
#import neologdn
import itertools
import unicodedata
import collections
import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
import streamlit as st
from pathlib import Path
import plotly.express as px
from st_aggrid import AgGrid
import plotly.graph_objs as go
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.utils import shuffle
from dataclasses import dataclass
import streamlit.components.v1 as components

load_dotenv(verbose=True)
MYREPOPATH = Path(os.environ.get("MYREPOPATH"))
FONT_TTF_PATH = Path(os.environ.get("FONT_TTF_PATH"))
USRUPLOADFILE_DIC_PATH = Path(os.environ.get("USRUPLOADFILE_DIC_PATH")) #ユーザーのアップロードファイルが格納されるディレクトリパス
UPLOADFORMAT_CSV_PATH = Path(os.environ.get("UPLOADFORMAT_CSV_PATH")) #アップロード用のフォーマットファイル
UPLOADFORMAT_WITHCATEGORY_CSV_PATH = Path(os.environ.get("UPLOADFORMAT_WITHCATEGORY_CSV_PATH"))
STOPWORD_TXT_PATH = Path(os.environ.get("STOPWORD_TXT_PATH")) #stopwordのファイルへのパス
REQUIRE_COLUMN_NAME = os.environ.get("REQUIRE_COLUMN_NAME")
UPLOADFILEMANAGE_CSV_PATH = Path(os.environ.get("UPLOADFILEMANAGE_CSV_PATH")) #アップロードされたファイル群を整理したファイルのパス
WORD_CONVERT_FORMAT_CSV_PATH = Path(os.environ.get("WORD_CONVERT_FORMAT_CSV_PATH")) #単語の変換用のフォーマットファイルへのパス
NODE_CRITERION_PAIRCOUNT_LENGTH_LIMIT = int(os.environ.get("NODE_CRITERION_PAIRCOUNT_LENGTH_LIMIT")) #300
DATA_DIRECTORY_PATH = Path(os.environ.get("DATA_DIRECTORY_PATH"))

sys.path.append(str(MYREPOPATH / "src"))
from QA_utils import get_stopword_list, check_not_None_and_False, make_sents_array, set_fileinfo, read_fileinfo_and_concat, fillna_df
from QA_natural_language_processing import ExeMorphologicalAnalysisWithWordconvertdict, ExeNLPPreprocess, add_pn_to_word_count_partofspeech_df

def questionnaire_uploader(require_column_name, upload_text: str):
    """
    ファイルをアップロードする。アップロードされてなければどこからかアップロードする？

    parameters
    ----------
    upload_text: str
        アップローダーの説明文

    Returns
    ----------
        uplaod_df: pd.DataFrame() or bool
        アップロードされたファイルが入っている。
    """
    uploadfile_path=st.file_uploader(upload_text, type="csv")
    if uploadfile_path:
        uploaded_raw_df = pd.read_csv(uploadfile_path)
        if require_column_name != uploaded_raw_df.columns[0]:
            st.error(f"アップロードされたファイルがフォーマットに従っていないようです。ファイルのA1セルには{require_column_name}が記入される必要があります。ファイルをアップロードし直してください。")
            uploaded_raw_df = None
        else:
            #pass
            uploaded_raw_df = fillna_df(uploaded_raw_df)
    else:
        uploaded_raw_df = False
    return uploaded_raw_df

def make_format_downloead_button(sample_data_path: str, button_msg, file_name:str):
    """
    ユーザーがファイルをアップロードするためのフォーマットファイルのダウンロードボタンの作成
    """

    sample_data = pd.read_csv(sample_data_path, index_col=0)
    sample_data = sample_data.to_csv().encode('utf-8_sig')
    st.download_button(
        label=button_msg,
        data=sample_data,
        file_name=file_name,
        mime='text/csv',
    )

def return_save_button(selected_project_name):
    """
    ファイル名が記入されていたらsave_buttonを出現させ、True or Falseを与える。
    """
    save_button_label = "保存する"
    SAMPLEDATA_FILE_NAME = "サンプルデータ"
    if selected_project_name: 
        USRUPLOADFILES = [_.stem for _ in USRUPLOADFILE_DIC_PATH.glob("*") if _.stem[0] != "."] #.で始まってないファイル・フォルダ
        # ユーザーが指定したファイル名がサンプルと同じか、過去に記録があるか
        if selected_project_name in USRUPLOADFILES:
            if selected_project_name == SAMPLEDATA_FILE_NAME:
                st.error("このファイル名は使用できません。他のファイル名を記入してください。")
                valid_selected_project_name_flag = False
            else:
                st.warning(f"現在指定しているファイル名は過去に利用されています。上書きしても良い場合は処理を続けてください。")
                valid_selected_project_name_flag = True
        
        elif "," in selected_project_name or "." in selected_project_name:
            st.error(",や.を含むファイル名は選択できません。ファイル名を変更してください。")
            valid_selected_project_name_flag = False
        else:
            valid_selected_project_name_flag = True
        # 上でvalidか判断し、ボタンを出現させる。
        if valid_selected_project_name_flag:
            save_button = st.button(label=save_button_label)
        else:
            save_button = False
    else:
        save_button = False

    return save_button

@dataclass
class StParametersFileUpload:
    number_remove_flag = False
    hiragana_remove_criteria_count = 1
    katakana_extract_flag = True
    adverb_remove_flag = False
    english_flag = False

    def set_number_remove_flag(self, side=False):
        number_remove_flag_text = "数字を取り除く（デフォルト：空白）：0, 1, ..., 9などの数字を取り除きます。"
        if side:
            self.number_remove_flag = st.sidebar.checkbox(number_remove_flag_text, value=False)
        else:
            self.number_remove_flag = st.checkbox(number_remove_flag_text, value=False)    
    
    def set_hiragana_remove_criteria_count(self, side=False):
        hiragana_remove_criteria_count_text = "「ひらがな」だけで構成された単語のうち、以下の文字数以下の単語を取り除く（デフォルト：1）：デフォルトの数値の場合、「する」、「やる」、「もっと」などの単語は取り除かれます。"
        if side:
            self.hiragana_remove_criteria_count = st.sidebar.number_input(hiragana_remove_criteria_count_text, 
            min_value=0, max_value=5, value=1, step=1) 
        else:
            self.hiragana_remove_criteria_count = st.number_input(hiragana_remove_criteria_count_text,
            min_value=0, max_value=5, value=1, step=1) 

    def set_katakana_extract_flag(self, side=False):
        katanaka_extract_flag_text = "文章中にカタカナが連続2文字以上続いたら単語として抽出する（デフォルト：チェック）：サクサクなどカタカナが連続する単語を強制的に単語として抽出します。"
        if side:
            self.katakana_extract_flag = st.sidebar.checkbox(katanaka_extract_flag_text, value=True)
        else:
            self.katakana_extract_flag = st.checkbox(katanaka_extract_flag_text, value=True)

    def set_adverb_remove_flag(self, side=False):
        adverb_remove_flag_text = "副詞を取り除く（デフォルト：空白）：副詞を取り除きます。"
        if side:
            self.adverb_remove_flag = st.sidebar.checkbox(adverb_remove_flag_text, value=False)
        else:
            self.adverb_remove_flag = st.checkbox(adverb_remove_flag_text, value=False)

    def set_english_flag(self, side=False):
        english_flag_text = "文章文章が英語の際にチェックをつけてください。"
        if side:
            self.english_flag = st.sidebar.checkbox(english_flag_text, value=False)
        else:
            self.english_flag = st.checkbox(english_flag_text, value=False)


def main():
    st.set_page_config(layout="wide")
    # パラメータの設定
    SPFU = StParametersFileUpload()
    page_number = "2"

    st.header(f"{page_number}. 分析したい文章のアップロード")
    st.markdown("""
    分析したい文章をアップロードします。
    下記の手順（2.1~2.4）に沿ってアップロードしてください。
    以前アップロードしたファイルを再度分析したい場合は、このページで操作を行わず、そのまま左のサイドバーの「ページ選択」から「ファイルの選択・管理」ページへ移動してください。
    """)
    
    #st.subheader(f"{page_number}.1 フォーマットのダウンロード")
    #st.markdown(f"""
    #文章をアプリにアップロードするためのフォーマットファイルを「ダウンロードボタン」を押下してダウンロードしてください。
    #フォーマットファイルは通常のフォーマットファイルと、カテゴリを追記できるフォーマットファイルの2種類があります。分析したい文章の形式に適したフォーマットをダウンロードしてください。
    #1. **通常のフォーマットファイル**：文章回答に対し、その回答者のカテゴリ（年代・出身地など）が**与えられてない**場合に利用するフォーマットファイル。
    #2. **カテゴリを追記できるフォーマットファイル**：文章回答に対し、その回答者のカテゴリ（年代・出身地など）が**与えられている**場合に利用するフォーマットファイル。
    #""")
    #st.subheader(f"""{page_number}.1.1 通常のフォーマットファイルのダウンロード""")
    #st.markdown(f"""
    #A行1列目のセルに表頭として{REQUIRE_COLUMN_NAME}が記載されていることを確認してください。ユーザーは1列目のB行以下に分析したい文章を貼り付けて、ファイルを上書き保存してください。
    #""")
    #button_msg = "ダウンロードボタン"
    #file_name = "テキストマイニング用フォーマットファイル.csv"
    #make_format_downloead_button(sample_data_path=UPLOADFORMAT_CSV_PATH, button_msg=button_msg, file_name=file_name)
    
    #st.subheader(f"""{page_number}.1.2 カテゴリを追記できるフォーマットファイルのダウンロード""")
    #st.markdown(f"""
    #A行1列目のセルに表頭として{REQUIRE_COLUMN_NAME}が記載されていることを確認してください。ユーザーは1列目のB行以下に分析したい文章を貼り付けてください。
    #また、このフォーマットでは各回答に対し、回答者のカテゴリを付与できます。
    #フォーマットの形式に従って、A行2列目、A行3列名・・・に表頭としてカテゴリ名を記載の上、各B行以下に回答者のカテゴリを記載してください。
    #""")
    #button_msg = "ダウンロードボタン"
    #file_name = "テキストマイニング用フォーマットファイル_カテゴリ付き.csv"
    #make_format_downloead_button(sample_data_path=UPLOADFORMAT_WITHCATEGORY_CSV_PATH, button_msg=button_msg, file_name=file_name)
    
    st.subheader(f"{page_number}.1 フォーマットのダウンロード")
    st.markdown(f"""
    以下の手順でフォーマットファイルをダウンロードし、分析したい文章を貼り付けてください。
    1. 文章をアプリにアップロードするフォーマットファイルを下の「ダウンロードボタン」を押下してダウンロードしてください。
    2. ダウンロードしたフォーマットファイルの{REQUIRE_COLUMN_NAME}列にテキストマイニングを実施した文章を1セルに1文章が入るよう貼り付けてください。
    3. 各文章に紐づくカテゴリ（年齢・所属など）があれば、それ以降の列に入力できます。ない場合は列を削除してください。
    """)
    button_msg = "ダウンロードボタン"
    file_name = "テキストマイニング用フォーマットファイル.csv"
    make_format_downloead_button(sample_data_path=UPLOADFORMAT_WITHCATEGORY_CSV_PATH, button_msg=button_msg, file_name=file_name)

    st.subheader(f"{page_number}.2 記入済みフォーマットのアップロード")
    st.markdown(f"""
    {page_number}.1で用意したファイルを「Broese files」ボタンからアプリにアップロードしてください。
    """)
    # ファイルアップロード
    tmp_uploaded_raw_df = questionnaire_uploader(require_column_name=REQUIRE_COLUMN_NAME, 
                                            upload_text="「Browse files」ボタンを押下し、「2.1 フォーマットのダウンロード」で保存したファイルをアプリにアップロードしてください。")
    if check_not_None_and_False(tmp_uploaded_raw_df):
        st.success("ファイルのアップロードの準備に成功しました。")
    else:
        st.warning('ファイルをアップロードしてください。')

    st.subheader(f"{page_number}.3 ファイル名の入力")
    #st.markdown('')
    selected_project_name = st.text_input(label="ファイル名を入力してください。ファイル名はアプリ内のファイルの管理に利用されます。", value="")
    if len(selected_project_name) > 0:
        if check_not_None_and_False(tmp_uploaded_raw_df):
            st.success("ファイル名を決定しました。")
        else:
            st.success(f"ファイル名を決定しました。しかし、ファイルがアップロードされていません。文章をアップロードしてください。")
    else:
        st.warning(f"ファイル名を入力し、再度Enterを押してください。この表示が黄色の場合はファイル名を決定できていません。名称が決定されると緑色に表示が変わります。")
    
    st.subheader(f"{page_number}.4 ファイルの保存")
    st.markdown(f"""
    {page_number}.3までの手順が正しく行われると下に「保存する」ボタンが表示されます。ファイルをアプリにアップロードする場合は「保存する」ボタンを押下してください。
    ボタンが押下されると、ファイルが保存され、同時にテキストマイニングの処理が起動します。処理には多少の時間がかかるので（1分以内）、その間ファイルを閉じないでください。
    """)
    #SPFU.set_english_flag()
    save_button = return_save_button(selected_project_name)

    #自然言語処理の前処理
    if save_button:
        if check_not_None_and_False(tmp_uploaded_raw_df):
            if len(selected_project_name) > 0:
                with st.spinner("""ファイルをアップロードしています。アプリを閉じたり、違うページへ移動しないでください。"""):
                    usr_directory_path = USRUPLOADFILE_DIC_PATH / selected_project_name #ユーザーがファイルをアプロードするためのディレクトリのパス
                    upload_raw_df_path = usr_directory_path / "raw_df.csv"
                    words_in_sent_df_csv_path = usr_directory_path / "words_in_sent_df.csv"
                    word_count_partofspeech_df_csv_path = usr_directory_path / "word_count_partofspeech_df.csv"
                    jaccard_df_csv_path = usr_directory_path / "jaccard_df.csv"
                    fileinfo_save_txt_path = usr_directory_path / "fileinfo.txt"
                    user_set_word_convert_path = usr_directory_path / "user_set_word_convert.csv"
                    user_processed_upload_path = usr_directory_path / "user_processed_upload.csv"
                    pn_noun_df_path = DATA_DIRECTORY_PATH / "dictionary/pn_noun_dict.csv"
                    pn_declinable_df_path = DATA_DIRECTORY_PATH / "dictionary/pn_declinable_dict.csv"

                    # 同じ名前のディレクトリが既にあった場合は一旦そのディレクトリを削除
                    if usr_directory_path.exists():
                        shutil.rmtree(usr_directory_path)
                    else:
                        pass
                    usr_directory_path.mkdir() #ユーザー指定のディレクトリを作成

                    # 保存
                    tmp_uploaded_raw_df.to_csv(upload_raw_df_path)
                    tmp_uploaded_raw_df.to_csv(user_processed_upload_path)
                    # 再read
                    reuploaded_raw_df = pd.read_csv(upload_raw_df_path, index_col=0)
                    reuploaded_raw_df = fillna_df(reuploaded_raw_df)

                    # 形態素解析対象を抽出
                    sents_array = make_sents_array(reuploaded_raw_df)

                    # 英語の時
                    if SPFU.english_flag:
                        ...
                    # 日本語のとき
                    else: 
                        #ストップワードの指定
                        stopword_list = get_stopword_list(STOPWORD_TXT_PATH)
                        #自然言語処理実施
                        EMAWW = ExeMorphologicalAnalysisWithWordconvertdict(tagger=MeCab.Tagger('-Ochasen'), 
                                                                            stopword_list=stopword_list, 
                                                                            word_convert_dict=None, 
                                                                            StParameters=SPFU)
                        enp = ExeNLPPreprocess(sents_array=sents_array)
                        words_in_sent_df, word_count_partofspeech_df, jaccard_df = enp.exe_nlp_preprocess(EMAWW)


                    #ポジネガの付与
                    pn_noun_df = pd.read_csv(pn_noun_df_path, index_col=0)
                    pn_declinable_df = pd.read_csv(pn_declinable_df_path, index_col=0)
                    pn_noun_dict = {tmp_word: tmp_pn for tmp_word, tmp_pn in zip(pn_noun_df["word"].values, pn_noun_df["pn"].values)}
                    pn_declinable_dict = {tmp_word: tmp_pn for tmp_word, tmp_pn in zip(pn_declinable_df["word"].values, pn_declinable_df["pn"].values)}
                    word_count_partofspeech_df = add_pn_to_word_count_partofspeech_df(word_count_partofspeech_df=word_count_partofspeech_df,
                                                                                      pn_noun_dict=pn_noun_dict,
                                                                                      pn_declinable_dict=pn_declinable_dict)

                    #保存
                    words_in_sent_df.to_csv(words_in_sent_df_csv_path)
                    word_count_partofspeech_df.to_csv(word_count_partofspeech_df_csv_path)
                    jaccard_df.to_csv(jaccard_df_csv_path)
                    shutil.copyfile(WORD_CONVERT_FORMAT_CSV_PATH, user_set_word_convert_path)
                    
                    #アップロード日,文章数,前処理の有無
                    set_fileinfo(target_raw_df=reuploaded_raw_df, 
                                fileinfo_save_txt_path=fileinfo_save_txt_path, 
                                file_name=selected_project_name,
                                preprocessing_or_not=False)

                    # fileinfoを読み取り、結合、保存
                    fileinfo_concat_df = read_fileinfo_and_concat(USRUPLOADFILE_DIC_PATH)
                    fileinfo_concat_df.to_csv(UPLOADFILEMANAGE_CSV_PATH)

                #成功表示
                st.success(f"{selected_project_name}というプロジェクト名で保存に成功しました。保存したファイルは「03 アップロードファイルの確認」から確認してください。")

            elif len(selected_project_name) == 0:
                st.error("保存するファイル名を決定した後、保存するボタンを押してください。")
            else:
                raise ValueError("管理者をよんでください。")
        else:
            st.error(f"ファイルをアップロードした後、保存してください。")
    else:
        #st.warning(f"ファイルをアップロードする準備が完了していないようです。")
        pass

if __name__ == "__main__":
    main()