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
REQUIRE_COLUMN_NAME = os.environ.get("REQUIRE_COLUMN_NAME") #sentence
UPLOADFILEMANAGE_CSV_PATH = Path(os.environ.get("UPLOADFILEMANAGE_CSV_PATH")) #アップロードされたファイル群を整理したファイルのパス
WORD_CONVERT_FORMAT_CSV_PATH = Path(os.environ.get("WORD_CONVERT_FORMAT_CSV_PATH")) #単語の変換用のフォーマットファイルへのパス

sys.path.append(str(MYREPOPATH / "src"))
from QA_utils import *
from QA_visualization import *

page_number = "3"
limit_level_num=50

def main():
    st.set_page_config(layout="wide")
    sorted_col_words_in_sent_df = ["words_in_sent", "index"]
    STAG = StParametersAgGrid()
    #STAG.set_AgGrid_color_map(side=True)
    
    st.header(f"{page_number}. ファイル・抽出単語の確認")
    st.markdown("""
    このページではアップロードしたファイル・そこから抽出された単語の一覧を確認・ダウンロードができます。
    """)
    st.subheader(f"{page_number}.1 アップロードされたファイルの一覧")
    st.markdown("""アップロードされているファイルは以下の通りです。""")
    uploadfile_manage_df = pd.read_csv(UPLOADFILEMANAGE_CSV_PATH)
    AgGrid(uploadfile_manage_df, theme=STAG.AgGrid_color_map)

    st.subheader(f"{page_number}.2 内容を確認したいファイルの選択")
    selected_project_name = select_questionnairefilename_from_management_files(file_manage_path=UPLOADFILEMANAGE_CSV_PATH, 
                                                                                labeltext="詳細を確認したいファイルを以下から選択してください。")

    #ファイルの読み込み
    questionnaire_raw_df, words_in_sent_df, word_count_partofspeech_df, jaccard_df = import_nlprocessed_files(UPLOADFILEMANAGE_CSV_PATH=USRUPLOADFILE_DIC_PATH, 
                                                                                                            selected_project_name=selected_project_name, 
                                                                                                            only_raw_df=False)

    st.subheader(f"{page_number}.3 選択したファイルの表示")
    st.markdown("アップロードした元ファイルを表示します。")
    questionnaire_raw_df[REQUIRE_COLUMN_NAME] = [f"{str(_)[:30]}..." if len(str(_)) > 30 else str(_) for _ in questionnaire_raw_df[REQUIRE_COLUMN_NAME].values]
    AgGrid(questionnaire_raw_df, theme=STAG.AgGrid_color_map)
    
    #DFの可視化
    st.subheader(f"{page_number}.4 文章から抽出された単語の表示")
    st.markdown("ツールが抽出した一文中の単語を表示します。右にスクロールすると抽出元の文章が表示されます。")
    #AgGrid(words_in_sent_df.loc[:, sorted_col_words_in_sent_df])
    words_in_sent_aggrid = words_in_sent_df.reset_index().loc[:, sorted_col_words_in_sent_df]
    words_in_sent_aggrid.columns = ["抽出された単語", "抽出元の文章"]
    words_in_sent_aggrid["抽出された単語"] = [f"{str(_)[:30]}..." if len(str(_)) > 30 else str(_) for _ in words_in_sent_aggrid["抽出された単語"].values]
    AgGrid(words_in_sent_aggrid, theme=STAG.AgGrid_color_map)

    # ダウンロード
    words_in_sent_df_download_button = st.download_button(
                                            label="ダウンロードボタン",
                                            data=words_in_sent_df.to_csv().encode('utf-8_sig'),
                                            file_name=f'文章中の単語_{selected_project_name}.csv',
                                            mime='text/csv',
                                        )


    st.subheader(f"{page_number}.5 抽出された単語一覧の表示")
    st.markdown("文章中に出現した単語を抽出・集計し、出現頻度が大きい順に表示します。")
    word_count_partofspeech_aggrid = copy.deepcopy(word_count_partofspeech_df)
    word_count_partofspeech_aggrid = word_count_partofspeech_aggrid.reset_index()
    if len(word_count_partofspeech_aggrid.columns) == 5:
        word_count_partofspeech_aggrid.columns = ["単語", "品詞", "出現数", "正式な品詞", "ポジネガ"]
    elif len(word_count_partofspeech_aggrid.columns) == 4:
        word_count_partofspeech_aggrid.columns = ["単語", "品詞", "出現数", "正式な品詞"]
    else:
        raise ValueError(f"word_count_partofspeech_aggridのカラム数が想定していないものが来ています（カラム数：{len(word_count_partofspeech_df.columns)}）。")
    AgGrid(word_count_partofspeech_aggrid, theme=STAG.AgGrid_color_map)

    # ダウンロード
    word_count_partofspeech_df_download_button = st.download_button(
                                            label="ダウンロードボタン",
                                            data=word_count_partofspeech_aggrid.to_csv().encode('utf-8_sig'),
                                            file_name=f'単語一覧_{selected_project_name}.csv',
                                            mime='text/csv',
                                        )
    
    st.subheader(f"{page_number}.6 回答のカテゴリの円グラフ")
    st.markdown("""文章に紐づいたカテゴリの割合を円グラフとして表示します。""")
    target_category_name = st.selectbox("カテゴリ名の選択", questionnaire_raw_df.columns[1:])
    category_count_df = pd.DataFrame(questionnaire_raw_df.loc[:, target_category_name].value_counts()).reset_index()
    category_count_df.columns = ["カテゴリ名", "カテゴリごとの出現数"]

    if len(questionnaire_raw_df.loc[:, target_category_name].unique()) < limit_level_num:
        SPC = StParametersClustering()
        SPC.set_pie_sequential_color_name()
        category_pie_fig = visualize_pie_chart(questionnaire_raw_df, target_category_name, SPC)
        st.plotly_chart(category_pie_fig)
        AgGrid(category_count_df)
    else:
        st.warning(f"カテゴリ内の水準数が{str(limit_level_num)}以上で多すぎるため円グラフの表示を停止しています。")
        AgGrid(category_count_df)

    st.subheader(f"{page_number}.7 文章の長さのヒストグラム")
    sentence_length_hist = visualize_hist([len(each_sent) for each_sent in questionnaire_raw_df["sentence"]], "文章長")
    st.plotly_chart(sentence_length_hist)

if __name__ == "__main__":
    main()