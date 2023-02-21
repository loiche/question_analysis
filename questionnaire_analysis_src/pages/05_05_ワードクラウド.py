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
import matplotlib.cm as cm
import plotly.express as px
from st_aggrid import AgGrid
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.utils import shuffle
from dataclasses import dataclass
import matplotlib.colors as mcolors
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
from QA_utils import *
from QA_natural_language_processing import ExeMorphologicalAnalysisWithWordconvertdict, ExeNLPPreprocess
from QA_wordcloud import StParametersWordcloud, make_wordcloud


@dataclass
class WordcloudPage:
    def main(self):
        st.markdown("""
        このページではアンケート結果をワードクラウドを用いて可視化します。
        ワードクラウドとは出現回数が多い単語ほど文字サイズを大きくして描画する可視化する手法です。
        ワードクラウドによってアンケート概要を簡単に把握することができます。
        """)
        st.subheader("5.1 ファイルの選択")
        selected_project_name = select_questionnairefilename_from_management_files(file_manage_path=UPLOADFILEMANAGE_CSV_PATH, 
                                                                                labeltext="ファイルを選択してください。")
        #ファイルの読み込み
        questionnaire_raw_df, words_in_sent_df, word_count_partofspeech_df, jaccard_df = import_nlprocessed_files(UPLOADFILEMANAGE_CSV_PATH=USRUPLOADFILE_DIC_PATH, 
                                                                                                                  selected_project_name=selected_project_name, 
                                                                                                                  only_raw_df=False)

        SPW = StParametersWordcloud()

        #ワードクラウドの描画
        st.subheader("5.2 ワードクラウド")
        st.sidebar.subheader("ワードクラウドのレイアウト設定")
        #SPW.set_wordcloud_parameters()
        SPW.set_prefer_horizontal_ratio(side=True)
        SPW.set_wordcloud_max_words(side=True)
        SPW.set_wordcloud_font_size(side=True)
        SPW.set_wordcloud_color_map(side=True)
        SPW.set_wordcloud_background_color(side=True)
        #SPW.set_wordcloud_randomstate()

        #wordcloud 
        #jaccardに合わせるか？
        whole_word_list = []
        for k, v in word_count_partofspeech_df["count"].items():
            for _ in range(v):
                whole_word_list.append(k)
        wordcloud_array = make_wordcloud(whole_word_list=whole_word_list, 
                                        font_path=str(FONT_TTF_PATH), 
                                        StParametersWordcloud=SPW,
                                        #tmp_wordcount_dict=word_count_partofspeech_df["count"] #色付けに利用
                                        )
        st.image(wordcloud_array)

        st.subheader("単語一覧の表示")
        st.markdown("文章中の単語を集計し、出現頻度が大きい順に表示しています。出現頻度の高い単語がワードクラウドで大きく表示されていることを確認してください。")
        STAG = StParametersAgGrid()
        STAG.set_AgGrid_color_map(side=True)
        word_count_partofspeech_aggrid = copy.deepcopy(word_count_partofspeech_df)
        word_count_partofspeech_aggrid = word_count_partofspeech_aggrid.reset_index()
        if len(word_count_partofspeech_aggrid.columns) == 5:
            word_count_partofspeech_aggrid.columns = ["単語", "品詞", "出現数", "正式な品詞", "ポジネガ"]
        elif len(word_count_partofspeech_aggrid.columns) == 4:
            word_count_partofspeech_aggrid.columns = ["単語", "品詞", "出現数", "正式な品詞"]
        else:
            raise ValueError(f"word_count_partofspeech_aggridのカラム数が想定していないものが来ています（カラム数：{len(word_count_partofspeech_df.columns)}）。")
        AgGrid(word_count_partofspeech_aggrid, theme=STAG.AgGrid_color_map)


def main():
    st.set_page_config(layout="wide")
    #可視化
    st.header("5 ワードクラウド")
    WP = WordcloudPage()
    WP.main()

if __name__ == "__main__":
    main()