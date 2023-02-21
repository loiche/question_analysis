import re
import os
import sys
import copy
import MeCab
import mlask
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
from QA_utils import *
from QA_natural_language_processing import ExeMorphologicalAnalysisWithWordconvertdict, ExeNLPPreprocess
#from QA_wordcloud import make_wordcloud, StParametersWordcloud, make_wordcloud_nochache
from QA_visualization import StParametersClustering, visualize_pie_chart

@dataclass
class PosinegaAnalysis:
    def main(self):
        ...

def main():
    page_number = "10"
    sentence_length_limit_to_show = 50
    st.set_page_config(layout="wide")
    st.title(f"{page_number}. ポジネガ分析")
    st.markdown("""
    文章の内容がポジティブかネガティブかを自動で判断します。
    文章中の単語のポジティブ表現、ネガティブ表現を抽出し、それぞれの出現数の多数決で文章がポジティブかネガティブか判定します。
    """)

    st.subheader(f"{page_number}.1 ファイルの選択")
    # プロジェクトの指定
    selected_project_name = select_questionnairefilename_from_management_files(file_manage_path=UPLOADFILEMANAGE_CSV_PATH, 
                                                                                labeltext="ファイルを選択してください。")

    subheader_2 = f"{page_number}.2 ポジネガ分析の結果"
    subheader_3 = f"{page_number}.3 文章のポジネガの割合の円グラフ"
    subheader_4 = f"{page_number}.4 ポジネガ分析の結果のダウンロード"
    
    #ファイルのインポート
    questionnaire_raw_df, words_in_sent_df, word_count_partofspeech_df, jaccard_df = import_nlprocessed_files(UPLOADFILEMANAGE_CSV_PATH=USRUPLOADFILE_DIC_PATH, 
                                                                                                                selected_project_name=selected_project_name, 
                                                                                                                only_raw_df=False)
    # naを埋める
    questionnaire_raw_df = fillna_df(questionnaire_raw_df)
    words_in_sent_df = fillna_df(words_in_sent_df)
    sents_array = make_sents_array(questionnaire_raw_df)

    if "ポジネガ" in word_count_partofspeech_df.columns:
        word_pozinega_dict = dict(word_count_partofspeech_df["ポジネガ"])
        word_pozinega_key_set = set(word_pozinega_dict.keys())
        posinega_score_dict = {"sentence": words_in_sent_df.index, "ポジネガ": [], "スコア": [], "ポジティブワード":[], "ネガティブワード": []}

        for each_sent in words_in_sent_df["words_in_sent"].values:
            posinega_score = 0
            positive_words = ""
            negative_words = ""
            for each_word in each_sent.split(", "):
                if each_word in word_pozinega_key_set:
                    posinega_score += word_pozinega_dict[each_word]
                    if word_pozinega_dict[each_word] > 0:
                        positive_words += f"{each_word}, "
                    elif word_pozinega_dict[each_word] < 0:
                        negative_words += f"{each_word}, "
                else:
                    pass
            
            if posinega_score < 0:
                posinega_result = "negative"
            elif posinega_score == 0:
                posinega_result = "neutral"
            else:
                posinega_result = "positive"
                
            posinega_score_dict["ポジネガ"].append(posinega_result)
            posinega_score_dict["スコア"].append(posinega_score)
            posinega_score_dict["ポジティブワード"].append(positive_words[:-2])
            posinega_score_dict["ネガティブワード"].append(negative_words[:-2])
        posinega_score_df = pd.DataFrame(posinega_score_dict)

        posinega_score_df_download = posinega_score_df.set_index("sentence")
        posinega_score_df["sentence"] = [f"{str(_)[:30]}..." if len(str(_)) > 30 else str(_) for _ in posinega_score_df["sentence"].values]
        st.subheader(subheader_2)
        st.markdown("""
        - ポジネガ：文章のポジネガ予測の結果
        - スコア：ポジティブなワードの数 - ネガティブなワードの数。正の場合はポジティブ、負の場合はネガティブと判定する。
        - ポジティブワード：文章中のポジティブと判定された単語
        - ネガティブワード：文章中のネガティブと判定された単語
        """)
        AgGrid(posinega_score_df)
            
        SPC = StParametersClustering()
        SPC.set_pie_sequential_color_name(side=True)
        pie_fig = visualize_pie_chart(posinega_score_df, pie_label="ポジネガ", SP=SPC)

        st.subheader(subheader_3)
        st.plotly_chart(pie_fig, use_container_width=False)
        
        st.subheader(subheader_4)
        st.markdown("ポジネガ分析の結果をダウンロードする際は以下をクリックください。")
        download_button_for_dataframe(posinega_score_df_download, 
                                    download_file_name=f"{selected_project_name}_ポジネガ分析.csv")
    else:
        st.warning("ファイルがポジネガ分析に対応していません。違うファイルを選択するか、ファイルをアップロードしなおしてください。")


if __name__ == "__main__":
    main()