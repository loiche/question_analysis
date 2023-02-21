import re
import os
import sys
import copy
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
STOPWORD_TXT_PATH = Path(os.environ.get("STOPWORD_TXT_PATH")) #stopwordのファイルへのパス
REQUIRE_COLUMN_NAME = os.environ.get("REQUIRE_COLUMN_NAME") #sentece
UPLOADFILEMANAGE_CSV_PATH = Path(os.environ.get("UPLOADFILEMANAGE_CSV_PATH")) #アップロードされたファイル群を整理したファイルのパス
WORD_CONVERT_FORMAT_CSV_PATH = Path(os.environ.get("WORD_CONVERT_FORMAT_CSV_PATH")) #単語の変換用のフォーマットファイルへのパス
DATA_DIRECTORY_PATH = Path(os.environ.get("DATA_DIRECTORY_PATH"))

sys.path.append(str(MYREPOPATH / "src"))
from QA_utils import *
from QA_natural_language_processing import ExeMorphologicalAnalysisWithWordconvertdict, ExeNLPPreprocess, add_pn_to_word_count_partofspeech_df

henkanmoto = "置換元"
henkansaki = "置換先"

@dataclass
class StParametesUserProcessing:
    number_remove_flag = False
    hiragana_remove_criteria_count = 1
    katakana_extract_flag = True
    adverb_remove_flag = False
    english_flag = False

    def set_number_remove_flag(self, side=False):
        number_remove_flag_text = "数字を取り除く（デフォルト：空白）：チェックを入れると0, 1, ..., 9などの数字を取り除きます。"
        if side:
            self.number_remove_flag = st.sidebar.checkbox(number_remove_flag_text, value=False)
        else:
            self.number_remove_flag = st.checkbox(number_remove_flag_text, value=False)

    def set_adverb_remove_flag(self, side=False):
        adverb_remove_flag_text = "副詞を取り除く（デフォルト：空白）：チェックを入れると副詞を取り除きます。"
        if side:
            self.adverb_remove_flag = st.sidebar.checkbox(adverb_remove_flag_text, value=False)
        else:
            self.adverb_remove_flag = st.checkbox(adverb_remove_flag_text, value=False)
    
    def set_hiragana_remove_criteria_count(self, side=False):
        hiragana_remove_criteria_count_text = "「ひらがな」だけで構成された単語のうち、以下の文字数以下の単語を取り除く（デフォルト：1）"
        if side:
            self.hiragana_remove_criteria_count = st.sidebar.number_input(hiragana_remove_criteria_count_text, 
            min_value=0, max_value=5, value=self.hiragana_remove_criteria_count, step=1) 
        else:
            self.hiragana_remove_criteria_count = st.number_input(hiragana_remove_criteria_count_text,
            min_value=0, max_value=5, value=self.hiragana_remove_criteria_count, step=1) 

    def set_katakana_extract_flag(self, side=False):
        katanaka_extract_flag_text = "文章中にカタカナが連続2文字以上続いたら単語として抽出する（デフォルト：チェック）：サクサクなどカタカナが連続する単語を強制的に単語として抽出します。"
        if side:
            self.katakana_extract_flag = st.sidebar.checkbox(katanaka_extract_flag_text, value=True)
        else:
            self.katakana_extract_flag = st.checkbox(katanaka_extract_flag_text, value=True)

    def set_english_flag(self, side=False):
        english_flag_text = "アンケート文章が英語の際にチェックをつけてください。"
        if side:
            self.english_flag = st.sidebar.checkbox(english_flag_text, value=False)
        else:
            self.english_flag = st.checkbox(english_flag_text, value=False)


def download_word_convert_format(word_convert_dict_format_path: str):
    """
    word_convert_dict_format_path = WORD_CONVERT_FORMAT_CSV_PATH
    """
    #文字置換用のフォーマットのダウンロード
    word_convert_dict_format = pd.read_csv(word_convert_dict_format_path, index_col=0)
    word_convert_dict_format = word_convert_dict_format.to_csv().encode('utf-8_sig')
    #ダウンロードボタンの設置
    st.download_button(
        label="文字置換用フォーマットダウンロードボタン",
        data=word_convert_dict_format,
        file_name='文字置換用フォーマット.csv',
        mime='text/csv',
    )

def set_word_convert_dict(word_convert_path, labeltext="", streamlit_flag=True):
    if streamlit_flag:
        #ユーザー記入済み置換フォーマットのアップロード
        word_convert_dict_df = st.file_uploader(labeltext, type="csv")
        #アップロードを検知したら以下を実施。
        if check_not_None_and_False(word_convert_dict_df):
            word_convert_dict_df = pd.read_csv(word_convert_dict_df)
            #アップロードされたファイルがフォーマットを守っていたら、以下を実施。
            if henkanmoto in word_convert_dict_df.columns \
            and henkansaki in word_convert_dict_df.columns \
            and len(word_convert_dict_df.columns) == 2:
                st.success("置換用のファイルのアップロードに成功しました。")
                word_convert_dict = {row[henkanmoto]:row[henkansaki] for ix, row in word_convert_dict_df.iterrows()}
            else:
                st.error("ファイルの形式が想定されたものと異なります。フォーマットをダウンロードして、再度置換用ファイルを作成し直してください。")
                word_convert_dict = None
        else:
            st.warning("文字の置換を行う場合はファイルをアップロードしてください。")
            word_convert_dict_df = pd.read_csv(word_convert_path)
            word_convert_dict = {row[henkanmoto]:row[henkansaki] for ix, row in word_convert_dict_df.iterrows()}

    else:
        word_convert_dict = {"貴人": "きじん", "いちゅう": "井中", "世界":"せかい"}

    return word_convert_dict

def convert_valid_wordconvertdict(word_convert_dict):
    """
    word_convert_dictにNoneや数字が入っているとエラーになるので、数字は文字列に、Noneは''に変換する
    """
    valid_word_convert_dict = {}
    for tmp_henkanmoto, tmp_henkansaki in word_convert_dict.items():
        if tmp_henkanmoto is None and tmp_henkansaki is None:
            pass
        elif tmp_henkanmoto is None and tmp_henkansaki is not None:
            valid_word_convert_dict[""] = str(tmp_henkansaki)
        elif tmp_henkanmoto is not None and tmp_henkansaki is None:
            valid_word_convert_dict[str(tmp_henkanmoto)] = ""
        elif tmp_henkanmoto is not None and tmp_henkansaki is not None:
            valid_word_convert_dict[str(tmp_henkanmoto)] = str(tmp_henkansaki)
        else:
            st.error("予期せぬ辞書が登録されているようです。アップロードするファイルを変えるか、管理者を読んでください。")
            
    if len(valid_word_convert_dict.keys()) <= 0:
        valid_word_convert_dict = None
    else:
        pass
                
    return valid_word_convert_dict

@dataclass
class ConvertWords:
    def convert_sent_with_word_convert_dict(self, sents_array, word_convert_dict):
        word_convered_sents_array = []
        for word_before_conversion, word_after_conversion in tqdm(word_convert_dict.items()):
            #1回目の走査が終わっていない
            if len(word_convered_sents_array) < len(sents_array):
                for sent_ix, target_sent in enumerate(sents_array):
                    #置換
                    converted_sent = re.sub(word_before_conversion, word_after_conversion, target_sent)
                    #記録
                    word_convered_sents_array.append(converted_sent)
            else:
                for sent_ix, target_sent in enumerate(word_convered_sents_array):
                    #置換
                    converted_sent = re.sub(word_before_conversion, word_after_conversion, target_sent)
                    #記録
                    word_convered_sents_array[sent_ix] = re.sub(word_before_conversion, word_after_conversion, target_sent)

        word_convered_sents_array = np.array(word_convered_sents_array)
        return word_convered_sents_array

@dataclass
class PostConvertWordsInSentDf(ConvertWords):
    def post_convert_words_in_sent_df(self, target_words_in_sent_df, word_convert_dict):
        tmp_words_in_sent_df = copy.deepcopy(target_words_in_sent_df)
        target_col = "words_in_sent"
        assert target_col in tmp_words_in_sent_df.columns

        #インデックスの置換
        tmp_words_in_sent_df.index = self.convert_sent_with_word_convert_dict(sents_array=tmp_words_in_sent_df.index, 
                                                                              word_convert_dict=word_convert_dict)
        #target_col列の置換
        tmp_words_in_sent_df[target_col] = self.convert_sent_with_word_convert_dict(sents_array=tmp_words_in_sent_df[target_col].values, 
                                                                                    word_convert_dict=word_convert_dict)
        return tmp_words_in_sent_df
    
@dataclass
class PostConvertWordCountPartofspeechDf(ConvertWords):
    def post_convert_word_count_partofspeech_df(self, target_word_count_partofspeech_df, word_convert_dict):
        count_col = "count"
        tmp_word_count_partofspeech_df = copy.deepcopy(target_word_count_partofspeech_df)
        
        #インデックスの置換
        tmp_word_count_partofspeech_df.index = self.convert_sent_with_word_convert_dict(sents_array=tmp_word_count_partofspeech_df.index, 
                                                                                        word_convert_dict=word_convert_dict)
        #置換した結果ダブったものの個数を足す
        unique_word_count_partofspeech_df = tmp_word_count_partofspeech_df.loc[~tmp_word_count_partofspeech_df.index.duplicated(), :]
        for key, row in tmp_word_count_partofspeech_df.loc[tmp_word_count_partofspeech_df.index.duplicated(), :].iterrows():
            unique_word_count_partofspeech_df.loc[key, count_col] = tmp_word_count_partofspeech_df.loc[key, :][count_col].sum()
    
        return unique_word_count_partofspeech_df
    
@dataclass
class PostConvertJaccardDf(ConvertWords):
    def post_convert_jaccard_df(self, target_jaccard_df, word_convert_dict):
        freq_pair_col = "freq_pair"
        freq_first_word_in_pair_col = "freq_first_word_in_pair"
        freq_second_word_in_pair_col = "freq_second_word_in_pair"
        jaccard_coef_col = "jaccard_coef"
        first_word_in_pair = "first_word_in_pair"
        second_word_in_pair = "second_word_in_pair"
        
        assert freq_pair_col in target_jaccard_df.columns
        assert freq_first_word_in_pair_col in target_jaccard_df.columns
        assert freq_second_word_in_pair_col in target_jaccard_df.columns
        assert jaccard_coef_col in target_jaccard_df.columns
        
        tmp_jaccard_df = copy.deepcopy(target_jaccard_df)
        
        #インデックスの置換
        tmp_jaccard_df.index = self.convert_sent_with_word_convert_dict(sents_array=tmp_jaccard_df.index, 
                                                                        word_convert_dict=word_convert_dict)
        
        print(type(word_convert_dict))
        
        # first_word_in_pair, second_word_in_pairの単語も置換
        tmp_jaccard_df[first_word_in_pair] = [word_convert_dict[_] if _ in word_convert_dict.keys() else _ for _ in tmp_jaccard_df[first_word_in_pair].values]
        tmp_jaccard_df[second_word_in_pair] = [word_convert_dict[_] if _ in word_convert_dict.keys() else _ for _ in tmp_jaccard_df[second_word_in_pair].values]
        
        #置換した結果ダブったものの個数を足す
        unique_jaccard_df = tmp_jaccard_df.loc[~tmp_jaccard_df.index.duplicated(), :]
        for key, row in tmp_jaccard_df.loc[tmp_jaccard_df.index.duplicated(), :].iterrows():
            unique_jaccard_df.loc[key, freq_pair_col] = tmp_jaccard_df.loc[key, freq_pair_col].sum()
            
            # jaccard係数に必要な項目の抽出
            freq_pair_num = unique_jaccard_df.loc[key, freq_pair_col]
            freq_first_word_in_pair_num = unique_jaccard_df.loc[key, freq_first_word_in_pair_col]
            freq_second_word_in_pair_num = unique_jaccard_df.loc[key, freq_second_word_in_pair_col]
            
            # 計算と上書き
            unique_jaccard_df.loc[key, jaccard_coef_col] = freq_pair_num / (freq_first_word_in_pair_num + freq_second_word_in_pair_num - freq_pair_num)
            
        # 変換によりfirst_word_in_pair, second_word_in_pairが一致したものを削除する。
        unique_jaccard_df = unique_jaccard_df.loc[list(unique_jaccard_df[first_word_in_pair] != unique_jaccard_df[second_word_in_pair]), :]
        
        #ソート
        unique_jaccard_df = unique_jaccard_df.sort_values(by=freq_pair_col, ascending=False)
        
        return unique_jaccard_df

def main():
    st.set_page_config(layout="wide")
    page_number = "4"
    st.header(f"{page_number}. 単語の置換・前処理の設定")
    sorted_col_words_in_sent_df = ["words_in_sent", "index"]
    st.markdown(f"""
    このページでは以下を実行できます。
    1. 単語の置換：表記揺れがある単語をユーザーが指定する単語に置換します。単語の置換は{page_number}.2から{page_number}.3までを設定してください。
    2. テキストマイニングの前処理の設定：テキストマイニングでは分析精度を上げるための前処理として「数字」や「副詞」、「ひらがなのみの単語」などを削除することがあります。前処理の設定は{page_number}.4で設定してください。
    いずれも設定後は{page_number}.5 の実行ボタンを押して、操作を実行してください。
    """)

    #"""テキストマイニングでは、アンケート回答中に**表記揺れ**が発生した場合、あるいは文章の複雑さに対してツールの**単語抽出精度が不足していた**場合など、ツールがアンケートから単語を正しく抽出できず、それゆえ解析が上手く実施できないことがあります。
    #このページでは、そうした問題に対し、「表記揺れが発生した単語」や「正しく抽出されなかった単語」を「ユーザーが指定する正しい単語」に置換できます。
    #表記揺れや抽出ミスを発見した場合は、このページで単語の置換を行なってください。
    #- 表記揺れ：漢字・送り仮名の違いにより同一単語であるにも関わらず、違う単語として認識されてしまう問題（「おいしい」と「美味しい」など）
    #- 単語の抽出精度不足：ツールの単語抽出精度が不十分であるため、本来の意味と違う単語として抽出されてしまう問題（「ジューシー」が「14」と検出されてしまう）。
    #"""

    #置換対象の決定
    st.subheader(f"{page_number}.1 ファイルの選択")
    selected_project_name = select_questionnairefilename_from_management_files(file_manage_path=UPLOADFILEMANAGE_CSV_PATH, 
                                                                                labeltext="ファイルを選択してください。")

    #ファイルの読み込み
    questionnaire_raw_df = import_nlprocessed_files(UPLOADFILEMANAGE_CSV_PATH=USRUPLOADFILE_DIC_PATH, 
                                                    selected_project_name=selected_project_name, 
                                                    only_raw_df=True)

    #array化
    sents_array = make_sents_array(questionnaire_raw_df)
    #ストップワードの指定
    stopword_list = get_stopword_list(STOPWORD_TXT_PATH)

    #置換準備
    st.subheader(f"{page_number}.2 文字置換用フォーマットのダウンロード")
    st.markdown("""
    置換用のフォーマットをダウンロードしてください。
    フォーマット中の「置換元」と「置換先」を記入すると、「置換元」の単語が「置換先」の単語に置換されます。
    「置換先」の単語を空白にしておくと、「置換元」の単語がアンケートから削除されます。
    デフォルトで「あじのもと」から「味の素」へ置換するファイルを例としてアップロードしています。
    """)
    download_word_convert_format(word_convert_dict_format_path=WORD_CONVERT_FORMAT_CSV_PATH)
    #ユーザー記入済み置換フォーマットのアップロード
    st.subheader(f"{page_number}.3 記入済み文字置換用フォーマットのアップロード")
    st.markdown(f"""
    {page_number}.2で記入した文字置換用フォーマットをアップロードしてください。
    """)
    word_convert_dict = set_word_convert_dict(word_convert_path=WORD_CONVERT_FORMAT_CSV_PATH, 
                                            labeltext="Browse filesボタンを押下して、記入済み文字置換用フォーマットをアップロードしてください。")

    #ファイルをアップロードするか否か
    if check_not_None_and_False(word_convert_dict):
        #ファイルの確認
        st.markdown("アップロードされたファイルを表示しています。「置換元」と「置換先」に間違いがないか確認してください。")
        word_convert_df = pd.DataFrame(word_convert_dict, index=["tmp_ix"]).T.reset_index()
        word_convert_df.columns = [henkanmoto, henkansaki]
        word_convert_dict = convert_valid_wordconvertdict(word_convert_dict)
    else:
        raise ValueError("_set_word_convert_dictを確認してください。")

    #アップロードファイルの確認
    STAG = StParametersAgGrid()
    STAG.set_AgGrid_color_map(side=True)
    AgGrid(word_convert_df, 
        #theme=STAG.AgGrid_color_map
        )
    
    st.subheader(f"{page_number}.4 自由記述文章への前処理の設定（β版）")
    st.markdown("""
    単語の抽出条件を変更したい場合は、以下の値を変更してください。
    """)
    SPUP = StParametesUserProcessing()
    SPUP.set_number_remove_flag()
    SPUP.set_adverb_remove_flag()
    SPUP.set_katakana_extract_flag()
    SPUP.set_hiragana_remove_criteria_count()

    st.subheader(f"{page_number}.5 単語の置換・前処理の実行")
    st.markdown(f"""
    「{selected_project_name}」に対し、ユーザーが設定した単語の置換、前処理を実行します。実行する場合は下のボタンを押してください。
    置換が完了するまで数秒の時間がかかります。処理が完了すると変更内容が表示されるので、それまでアプリを閉じたり、他のページへ移動しないでください。
    """)
    convert_button = st.button("前処理・単語の置換を実施する")

    if convert_button:
        # ファイル設定
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

        if len(word_convert_df) > 0:
            #ユーザーの辞書を保存
            word_convert_df.to_csv(user_set_word_convert_path)

        with st.spinner("""文字を置換しています。アプリを閉じたり、違うページへ移動しないでください。"""):
            #文字を置換し、置換先を保存する
            CW = ConvertWords()
            sents_array = CW.convert_sent_with_word_convert_dict(sents_array=sents_array, 
                                                                    word_convert_dict=word_convert_dict
                                                                )

            #ストップワードの指定
            stopword_list = get_stopword_list(STOPWORD_TXT_PATH)
            #自然言語処理実施 with 置換
            EMAWW = ExeMorphologicalAnalysisWithWordconvertdict(tagger=MeCab.Tagger('-Ochasen'), 
                                                                stopword_list=stopword_list, 
                                                                word_convert_dict=None, 
                                                                StParameters=SPUP)
            enp = ExeNLPPreprocess(sents_array=sents_array)
            words_in_sent_df, word_count_partofspeech_df, jaccard_df = enp.exe_nlp_preprocess(EMAWW)
            
            #uploadfileの上書き
            processed_questionnaire_raw_df = copy.deepcopy(questionnaire_raw_df)
            processed_questionnaire_raw_df[REQUIRE_COLUMN_NAME] = sents_array
            
            #words_in_sent_dfの後置換
            PCWISD = PostConvertWordsInSentDf()
            post_processed_words_in_sent_df = PCWISD.post_convert_words_in_sent_df(target_words_in_sent_df=words_in_sent_df, 
                                                                                word_convert_dict=word_convert_dict)
            #word_count_partofspeech_dfの後置換
            PCWCPD = PostConvertWordCountPartofspeechDf()
            post_processed_word_count_partofspeech_df=PCWCPD.post_convert_word_count_partofspeech_df(target_word_count_partofspeech_df=word_count_partofspeech_df, 
                                                                                                    word_convert_dict=word_convert_dict)
            # jaccard_dfの後置換
            PCJD = PostConvertJaccardDf()
            post_processed_jaccard_df = PCJD.post_convert_jaccard_df(target_jaccard_df=jaccard_df, 
                                                                    word_convert_dict=word_convert_dict)
    
            pn_noun_df = pd.read_csv(pn_noun_df_path, index_col=0)
            pn_declinable_df = pd.read_csv(pn_declinable_df_path, index_col=0)
            pn_noun_dict = {tmp_word: tmp_pn for tmp_word, tmp_pn in zip(pn_noun_df["word"].values, pn_noun_df["pn"].values)}
            pn_declinable_dict = {tmp_word: tmp_pn for tmp_word, tmp_pn in zip(pn_declinable_df["word"].values, pn_declinable_df["pn"].values)}
            post_processed_word_count_partofspeech_df = add_pn_to_word_count_partofspeech_df(word_count_partofspeech_df=post_processed_word_count_partofspeech_df,
                                                                                pn_noun_dict=pn_noun_dict,
                                                                                pn_declinable_dict=pn_declinable_dict)
            #保存
            processed_questionnaire_raw_df.to_csv(user_processed_upload_path)
            post_processed_words_in_sent_df.to_csv(words_in_sent_df_csv_path)
            post_processed_word_count_partofspeech_df.to_csv(word_count_partofspeech_df_csv_path)
            post_processed_jaccard_df.to_csv(jaccard_df_csv_path)
            
            #アップロード日,文章数,前処理の有無
            set_fileinfo(target_raw_df=questionnaire_raw_df, 
                        fileinfo_save_txt_path=fileinfo_save_txt_path, 
                        file_name=selected_project_name,
                        preprocessing_or_not=True)
            # fileinfoを読み取り、結合、保存
            fileinfo_concat_df = read_fileinfo_and_concat(USRUPLOADFILE_DIC_PATH)
            fileinfo_concat_df.to_csv(UPLOADFILEMANAGE_CSV_PATH)

        #メッセージ
        st.success("文字の置換に成功しました。")
        st.subheader("4.6 変換結果")
        st.markdown("変換後の文章からの抽出単語")
        AgGrid(post_processed_word_count_partofspeech_df.reset_index(), 
            #theme=STAG.AgGrid_color_map
            )

    else:
        pass
        #st.warning("置換の準備ができました。置換を実施する場合は「単語の置換を実施する」ボタンを押してください。") 



if __name__ == "__main__":
    main()