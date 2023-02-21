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

sys.path.append(str(MYREPOPATH / "src"))
from QA_utils import get_stopword_list, check_not_None_and_False, make_sents_array, pathlib_glob_without_dotstartfile, set_fileinfo, select_questionnairefilename_from_management_files, import_nlprocessed_files, read_fileinfo_and_concat, get_count_of_target_pattern, fillna_df
from QA_natural_language_processing import ExeMorphologicalAnalysisWithWordconvertdict, ExeNLPPreprocess
from QA_wordcloud import make_wordcloud, StParametersWordcloud, make_wordcloud_nochache

def make_candidates_words_list(word_count_partofspeech_df, word_length=50):
    assert "count" in word_count_partofspeech_df.columns
    candidates_words_list = word_count_partofspeech_df.head(word_length)["count"].index
    return candidates_words_list

@dataclass
class CategoryAnalysis:
    def main(self):
        st.title("8. 回答者カテゴリごとの集計（β版）")
        # プロジェクトの指定
        selected_project_name = select_questionnairefilename_from_management_files(file_manage_path=UPLOADFILEMANAGE_CSV_PATH, 
                                                                                   labeltext="ファイルを選択してください。")

        #ファイルのインポート
        questionnaire_raw_df, words_in_sent_df, word_count_partofspeech_df, jaccard_df = import_nlprocessed_files(UPLOADFILEMANAGE_CSV_PATH=USRUPLOADFILE_DIC_PATH, 
                                                                                                                  selected_project_name=selected_project_name, 
                                                                                                                  only_raw_df=False)
        # naを埋める
        questionnaire_raw_df = fillna_df(questionnaire_raw_df)
        words_in_sent_df = fillna_df(words_in_sent_df)
        sents_array = make_sents_array(questionnaire_raw_df)

        # 定数の指定
        feature_cal_method = "count"
        ranking_length = 30
        words_in_sent_col = "words_in_sent"
        limit_level_num = 31

        if len(sents_array) < 200:
            candidate_word_length = len(word_count_partofspeech_df)
        else:
            candidate_word_length = 300

        if len(questionnaire_raw_df.columns[1:]) == 0:
            st.error("アップロードしたファイルにカテゴリの登録がありません。")
        else:
            target_user_category_name = st.selectbox("カテゴリ名の選択", questionnaire_raw_df.columns[1:])
            if len(questionnaire_raw_df.loc[:, target_user_category_name].unique()) < limit_level_num:
                target_level = st.selectbox("カテゴリの選択", questionnaire_raw_df[target_user_category_name].unique())
                #target_category_freqwordname_dict = {}
                #target_category_freqwordcount_dict = {}
                #for target_user_level in tqdm(questionnaire_raw_df.loc[:, target_user_category_name].unique()):
                #    #水準に合致したファイルの抽出
                #    target_category_bool_list = list(questionnaire_raw_df.loc[:, target_user_category_name] == target_user_level)
                #    target_category_words_in_sent_df = words_in_sent_df.loc[target_category_bool_list, :]
                #    target_category_sents_array = sents_array[target_category_bool_list]
                #    
                #    #文章全体で出現頻度の高い単語300個を取り出す
                #    candidates_words_list = make_candidates_words_list(word_count_partofspeech_df, word_length=candidate_word_length)
                #    
                #    #ヒット単語の検索
                #    target_category_word_feature_df = get_count_of_target_pattern(target_category_words_in_sent_df[words_in_sent_col].values, 
                #                                                                target_category_sents_array,
                #                                                                candidates_words_list, 
                #                                                                feature_cal_method)
                #    #集計
                #    target_category_freqwordname_dict[target_user_level] = target_category_word_feature_df.sum().sort_values(ascending=False).head(ranking_length).keys()
                #    target_category_freqwordcount_dict[target_user_level] = target_category_word_feature_df.sum().sort_values(ascending=False).head(ranking_length).values
                #
                ## DF化
                #target_category_freqwordcount_df =  pd.DataFrame(target_category_freqwordcount_dict)
                #target_category_freqwordcount_df.columns = [f"{_}_出現数" for _ in target_category_freqwordcount_df.columns]
                #target_category_freqwordname_df = pd.DataFrame(target_category_freqwordname_dict)

                ## 表頭の設定
                #target_category_freqwordname_df.columns = [_ if _ != "" else "登録カテゴリなし" for _ in target_category_freqwordname_df.columns]
                #target_category_freqwordcount_df.columns = [_ if _ != "_出現数" else "登録カテゴリなし_出現数" for _ in target_category_freqwordcount_df.columns]
                #target_category_freqwordname_count_df = pd.concat([target_category_freqwordname_df, target_category_freqwordcount_df], axis=1)
                #target_category_freqwordname_count_df_columns = []
                #for _ in sorted(target_category_freqwordname_df.columns):
                #    target_category_freqwordname_count_df_columns.append(_)
                #    target_category_freqwordname_count_df_columns.append(f"{_}_出現数")
                ## 整列
                #target_category_freqwordname_count_df = target_category_freqwordname_count_df.loc[:, target_category_freqwordname_count_df_columns]

                ##表の表示
                #AgGrid(target_category_freqwordname_count_df)
                with st.spinner("""処理を開始しました。"""):
                    # 処理
                    target_level_questionnaire_df = questionnaire_raw_df.groupby(target_user_category_name).get_group(target_level)
                    target_level_words_in_sent_df = words_in_sent_df.iloc[target_level_questionnaire_df.index, :]
                    target_level_sents_array = make_sents_array(target_level_questionnaire_df)
                    
                    # 単語・単語ペアのカウント
                    target_level_word_count_dict = {}
                    target_level_word_pair_list = []
                    for _ in target_level_words_in_sent_df["words_in_sent"].values:
                        for tmp_word in _.split(", "):
                            if tmp_word in target_level_word_count_dict.keys():
                                target_level_word_count_dict[tmp_word] += 1
                            else:
                                target_level_word_count_dict[tmp_word] = 1

                        target_level_pair_words = list(itertools.combinations(_.split(", "), 2))
                        target_level_word_pair_list.extend(target_level_pair_words)
                        
                    # インスタンス化
                    enp = ExeNLPPreprocess(sents_array=target_level_sents_array)
                    
                    # jaccard係数計算の準備
                    target_level_whole_word_list = enp.convert_whole_word_part_of_speech_dict_to_whole_word_list(target_level_word_count_dict)
                    #jaccard係数を計算
                    target_level_jaccard_df = enp.count_word_pair_freq(target_level_word_pair_list, target_level_whole_word_list, target_level_word_count_dict)
                    #表示量を決める
                    target_level_jaccard_df = enp.extract_jaccard_df_freq_pair_top(target_level_jaccard_df)
                    


                    st.header("該当カテゴリの回答の表示")
                    sentence_length_limit_to_show = 20
                    target_level_questionnaire_df_aggrid = copy.deepcopy(target_level_questionnaire_df)
                    target_level_questionnaire_df_aggrid[REQUIRE_COLUMN_NAME] = [_[:sentence_length_limit_to_show] if _[:sentence_length_limit_to_show] == _ else f"{_[:sentence_length_limit_to_show]}..." for _ in target_level_questionnaire_df_aggrid[REQUIRE_COLUMN_NAME].values]
                    AgGrid(target_level_questionnaire_df_aggrid, theme="blue")
                    
                    
                    st.header("ワードクラウド")
                    # パラメータ設定
                    SPW = StParametersWordcloud()
                    #ワードクラウドの描画
                    st.markdown("""
                    ワードクラウドとは出現回数が多い単語ほど、文字サイズを大きくして描画する可視化する手法です。
                    """)
                    st.sidebar.subheader("ワードクラウドのレイアウト設定")
                    #SPW.set_wordcloud_parameters()
                    SPW.set_prefer_horizontal_ratio(side=True)
                    SPW.set_wordcloud_max_words(side=True)
                    SPW.set_wordcloud_font_size(side=True)
                    SPW.set_wordcloud_color_map(side=True)
                    SPW.set_wordcloud_background_color(side=True)
                    #SPW.set_wordcloud_randomstate()
                    
                    wordcloud_array = make_wordcloud_nochache(whole_word_list=target_level_whole_word_list, 
                                            font_path=str(FONT_TTF_PATH), 
                                            StParametersWordcloud=SPW,
                                            tmp_wordcount_dict=target_level_word_count_dict)
                    st.image(wordcloud_array)


                    st.header("該当カテゴリの回答の単語ランキング")
                    AgGrid(pd.DataFrame(target_level_word_count_dict, index=["count"]).T.reset_index().sort_values(by="count", ascending=False))    

                        
            else:
                st.error("カテゴリ内の水準数が多すぎます（30以上）。アプリが止まる可能性があるため処理を停止しました。")
                #large_level_error_message = """
                #カテゴリを以下に列挙します。
                #"""
                #for num, tmp_level in enumerate(questionnaire_raw_df.loc[:, target_user_category_name].unique()):
                #    large_level_error_message += f"""
                #    {str(num)}. {tmp_level}
                #    """
                #st.markdown(large_level_error_message)



def main():
    st.set_page_config(layout="wide")
    CA = CategoryAnalysis()
    CA.main()

if __name__ == "__main__":
    main()