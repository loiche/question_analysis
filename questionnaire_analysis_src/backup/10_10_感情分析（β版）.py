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
    """)

    st.subheader(f"{page_number}.1 ファイルの選択")
    # プロジェクトの指定
    selected_project_name = select_questionnairefilename_from_management_files(file_manage_path=UPLOADFILEMANAGE_CSV_PATH, 
                                                                                labeltext="ファイルを選択してください。")

    subheader_2 = f"{page_number}.2 ポジネガ分析の実行"
    subheader_3 = f"{page_number}.3 文章のポジネガの割合の円グラフ"
    subheader_4 = f"{page_number}.4 文章のポジネガ分析の結果と判断根拠となった単語"
    subheader_5 = f"{page_number}.5 ポジネガ分析の結果のダウンロード"   

    st.subheader(subheader_2)
    st.markdown("""ポジネガ分析を実行する場合は下の実行ボタンを押して下さい。実行には時間がかかります（1分以内）。実行中はアプリを閉じたり、違うページに移動しないようにしてください。実行中に操作を中断したい場合は再読み込みをしてください。""")
    exe_button = st.button(label="ポジネガ分析実行ボタン")

    SPC = StParametersClustering()
    
    if exe_button:
        with st.spinner():
            #ファイルのインポート
            questionnaire_raw_df, words_in_sent_df, word_count_partofspeech_df, jaccard_df = import_nlprocessed_files(UPLOADFILEMANAGE_CSV_PATH=USRUPLOADFILE_DIC_PATH, 
                                                                                                                        selected_project_name=selected_project_name, 
                                                                                                                        only_raw_df=False)
            # naを埋める
            questionnaire_raw_df = fillna_df(questionnaire_raw_df)
            words_in_sent_df = fillna_df(words_in_sent_df)
            sents_array = make_sents_array(questionnaire_raw_df)

            with st.spinner():
                emotion_dict = {}
                for target_sents in sents_array:
                    target_sent_str = str(target_sents)
                    for target_sent in target_sents.split("。"):
                        target_sent += "。"

                        if target_sent == "。":
                            continue

                        try:
                            emotion_dict[target_sent] = mlask.MLAsk().analyze(target_sent)
                        except:
                            st.error(f"{target_sent}のポジネガ分析にて予期せぬエラーが発生しました。{target_sent}を除いて分析を進めています。")
                emotion_df = pd.DataFrame(emotion_dict).T
                
                emotion_df["orientation"] = emotion_df["orientation"].fillna("NETURAL")
                pie_fig = visualize_pie_chart(emotion_df, pie_label="orientation", SP=SPC)

                st.subheader(subheader_3)
                st.plotly_chart(pie_fig, use_container_width=False)
                
                st.subheader(subheader_4)
                emotion_agg_df = shorten_ix_for_Aggrid(target_agg_df=emotion_df.loc[:, ["orientation", "representative"]], sentence_length_limit_to_show=sentence_length_limit_to_show)
                emotion_agg_df.columns = ["テキスト", "ポジネガ", "判断根拠の単語"]
                AgGrid(emotion_agg_df)
                
                st.subheader(subheader_5)
                st.markdown("ポジネガ分析の結果をダウンロードする際は以下をクリックください。")
                download_button_for_dataframe(emotion_agg_df, 
                                            download_file_name=f"{selected_project_name}_ポジネガ分析.csv")
    else:
        st.subheader(subheader_3)
        st.warning("実行ボタンを押すと表示されます。")
        st.subheader(subheader_4)
        st.warning("実行ボタンを押すと表示されます。")
        st.subheader(subheader_5)
        st.warning("実行ボタンを押すと表示されます。")

if __name__ == "__main__":
    main()