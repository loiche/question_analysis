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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import streamlit.components.v1 as components
from sklearn.feature_extraction.text import TfidfTransformer

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

@dataclass
class StParametersClustering:
    sequential_color_name_dict = {
        "Reds": px.colors.sequential.Reds_r,
        "Greens": px.colors.sequential.Greens_r,
        "Blues": px.colors.sequential.Blues_r,
        "Oranges": px.colors.sequential.Oranges_r,
        "viridis": px.colors.sequential.Viridis,
        }
    #clusteringの配色
    clustering_color = sequential_color_name_dict["Reds"]
    # クラスタ名に頻出単語を何個表示するか
    frequent_words_in_cluster_num = 5
    #pieのmargin
    pie_margin_top = 0
    pie_margin_bottom = 0
    pie_margin_left = 0
    pie_margin_right = 0
    # pie chartの色
    pie_sequential_color = sequential_color_name_dict["Reds"]
    #pieのサイズ
    pie_width = 600
    pie_height = 400

    def set_frequent_words_in_cluster_num(self, side=False):
        frequent_words_in_cluster_num_text = "出現回数が高い単語を何個までクラスタ名と表示するか。"
        if side:
            self.frequent_words_in_cluster_num = st.sidebar.number_input(frequent_words_in_cluster_num_text, 
            min_value=0, max_value=10, value=5, step=1) 
        else:
            self.frequent_words_in_cluster_num = st.number_input(frequent_words_in_cluster_num_text,
            min_value=0, max_value=10, value=5, step=1)
            
    def set_clustering_color(self, side=False):
        if side:
            sequential_color_name = st.sidebar.selectbox("配色を指定してください。",
                                                         self.sequential_color_name_dict.keys())
        else:
            sequential_color_name = st.selectbox("配色を指定してください。",
                                                 self.sequential_color_name_dict.keys())            
        self.clustering_color = self.sequential_color_name_dict[sequential_color_name]
        
    def set_pie_sequential_color_name(self, side=False):
        if side:
            pie_sequential_color_name = st.sidebar.selectbox("円グラフの配色を指定してください。",
                                                         self.sequential_color_name_dict.keys())
        else:
            pie_sequential_color_name = st.selectbox("円グラフの配色を指定してください。",
                                                 self.sequential_color_name_dict.keys())            
        self.pie_sequential_color = self.sequential_color_name_dict[pie_sequential_color_name]
        
    def set_pie_margin_dict(self, side=False):
        if side:
            self.pie_margin_top = st.sidebar.slider(label="円グラフのtop margin（デフォルト10）",
                                                        min_value=0,
                                                        max_value=100,
                                                        value=10)
            self.pie_margin_bottom = st.sidebar.slider(label="円グラフのbottom margin（デフォルト10）",
                                                        min_value=0,
                                                        max_value=100,
                                                        value=10)
            self.pie_margin_left = st.sidebar.slider(label="円グラフのleft margin（デフォルト10）",
                                                        min_value=0,
                                                        max_value=100,
                                                        value=10)
            self.pie_margin_right = st.sidebar.slider(label="円グラフのright margin（デフォルト10）",
                                                        min_value=0,
                                                        max_value=100,
                                                        value=10)          
        else:
            self.pie_margin_top = st.slider(label="円グラフのtop margin（デフォルト10）",
                                                        min_value=0,
                                                        max_value=100,
                                                        value=10)
            self.pie_margin_bottom = st.slider(label="円グラフのbottom margin（デフォルト10）",
                                                        min_value=0,
                                                        max_value=100,
                                                        value=10)
            self.pie_margin_left = st.slider(label="円グラフのleft margin（デフォルト10）",
                                                        min_value=0,
                                                        max_value=100,
                                                        value=10)
            self.pie_margin_right = st.slider(label="円グラフのright margin（デフォルト10）",
                                                        min_value=0,
                                                        max_value=100,
                                                        value=10)

    def set_pie_fig_size(self, side=False):
        if side:
            self.pie_width = st.sidebar.slider(label="pie width（デフォルト600）",
                                                    min_value=100,
                                                    max_value=1000,
                                                    value=600)
            self.pie_height = st.sidebar.slider(label="pie height（デフォルト400）",
                                                        min_value=100,
                                                        max_value=1000,
                                                        value=400)
        else:
            self.pie_width = st.slider(label="pie width（デフォルト600）",
                                                    min_value=100,
                                                    max_value=1000,
                                                    value=600)
            self.pie_height = st.slider(label="pie height（デフォルト400）",
                                                        min_value=100,
                                                        max_value=1000,
                                                        value=400)



def make_word_bag(word_count_partofspeech_df,  
                  noun_length,
                  adjective_length, 
                  verb_length):
    """
    名詞・動詞・形容詞の頻出単語をそれぞれ取得し、リストとして与える。
    """
    partofspeech_length_dict = {"adjective": adjective_length,
                                "noun": noun_length, 
                               "verb": verb_length}
    word_bag_list = []
    for tmp_ps, tmp_ps_length in partofspeech_length_dict.items():
        if tmp_ps_length > 0:
            word_count_targetps_df = word_count_partofspeech_df[word_count_partofspeech_df.loc[:, "part_of_speech"] == tmp_ps]
            word_bag_list.extend(word_count_targetps_df.sort_values(by="count", ascending=False).index[:tmp_ps_length])
    return word_bag_list

def visualize_kmeans_with_pcs(targetx, targety, sents_array, y_km, cluster_name_dict, StParameters):
    sentence_length_limit = 100
    figure_margin_dict = dict(b=0,
                l=0,
                r=0,
                t=0)
    fig = go.Figure()
    for tmp_cluster_num in set(y_km):
        target_cluster_ix = list(y_km == tmp_cluster_num)
        fig.add_trace(go.Scatter(x = targetx[target_cluster_ix],
                                y = targety[target_cluster_ix],
                                hoverinfo='text',
                                hovertext=[_[:sentence_length_limit] for _ in sents_array[target_cluster_ix]],
                                name=cluster_name_dict[tmp_cluster_num], 
                                mode = 'markers',
                                marker_symbol="circle",
                                marker_line_color="black",
                                marker_line_width=1,
                                marker=dict(
                                    #showscale=True,
                                    #reversescale=True,
                                    #colorscale="OrRd",
                                    color=StParameters.clustering_color[int(tmp_cluster_num / np.unique(y_km).max() * len(px.colors.sequential.OrRd)-0.5)],
                                    size=10,#10,
                                    opacity=0.8,
                                    #colorbar=dict(
                                    #    #thickness=15,
                                    #    #title='jaccard係数',
                                    #    xanchor='left',
                                    #    titleside='right'
                                    # )
                            )))
    fig.update_layout(
            width=800,
            height=400,
            margin=figure_margin_dict,
    )
    #fig.update_layout(legend=dict(x=0.01,
    #                          y=0.99,
    #                          xanchor='left',
    #                          yanchor='top',
    #                          orientation='h',
    #                          ))
    return fig

def cal_word_importance(tmp_word_feature_df, y_km, cluster_num):
    word_importance_df = {}
    for target_cluster in range(cluster_num):
        word_importance_df[f"クラスタ{str(target_cluster+1)}"] = tmp_word_feature_df.loc[y_km == target_cluster, :].sum().astype(int)
    word_importance_df = pd.DataFrame(word_importance_df).reset_index()
    return word_importance_df

#def make_candidates_words_list(word_count_partofspeech_df, word_length=50):
#    assert "count" in word_count_partofspeech_df.columns
#    candidates_words_list = word_count_partofspeech_df.head(word_length)["count"].index
#    return candidates_words_list
    
def make_cluster_name_dict(word_importance_df, y_km, StParameters):
    """
    {0: 'クラスタ1:影男,ふたり,見る,かれる,できる',
     1: 'クラスタ2:せる,ぼく,影男,ひとり,できる',
     2: 'クラスタ3:顔,見る,見える,せる,かれる'}
     のようなクラスタの数字に対して表示するクラスタを返す。
    """
    assert "index" in word_importance_df.columns
    cluster_name_dict = {}
    for tmp_cluster_num in np.unique(y_km):
        
        target_cluster_name = f"クラスタ{str(tmp_cluster_num+1)}"
        if tmp_cluster_num != y_km.max():
            target_cluster_frequent_words_array = word_importance_df.set_index("index").sort_values(by=target_cluster_name, ascending=False).head(StParameters.frequent_words_in_cluster_num).index
            cluster_name_dict[tmp_cluster_num] = f"{target_cluster_name}: {','.join(target_cluster_frequent_words_array)}"
        else:
            cluster_name_dict[tmp_cluster_num] = f"該当クラスタなし"
            
    return cluster_name_dict


def visualize_pie_chart(match_pattern_df, pie_label, SP): 
    figure_margin_dict = dict(b=SP.pie_margin_bottom,
                    l=SP.pie_margin_left,
                    r=SP.pie_margin_right,
                    t=SP.pie_margin_top)
    fig = go.Figure(data=[go.Pie(labels=match_pattern_df.loc[:, pie_label].value_counts().keys(),
                                 values=match_pattern_df.loc[:, pie_label].value_counts().values)])
    fig.update_traces(hoverinfo='label+percent', 
                      textinfo='value', 
                      #textfont_size=20,
                      marker=dict(colors=SP.pie_sequential_color, 
                                  #line=dict(color='#000000', 
                                  #          width=2)
                                 ))
    fig.update_layout(
        width=SP.pie_width,
        height=SP.pie_height,
        margin=figure_margin_dict,
    )
    return fig

def main():
    st.set_page_config(layout="wide")
    page_number="7"
    st.header(f"{page_number}. 単語の検索・集計")
    #表示する文章の長さ
    sentence_length_limit_to_show = 80
    st.markdown("""
        このページではアンケートに出現する単語の検索・集計を行います。
        検索したい単語を指定すると、自動でアンケート中から単語を検索・集計し、各アンケート回答における単語の出現数を表にまとめます。
        """)

    np.random.seed(seed=0)
    st.subheader(f"{page_number}.1 ファイルの選択")
    #分析ファイルの選択
    selected_project_name = select_questionnairefilename_from_management_files(file_manage_path=UPLOADFILEMANAGE_CSV_PATH, 
                                                                            labeltext="ファイルを選択してください。")
    #ファイルの読み込み
    questionnaire_raw_df, words_in_sent_df, word_count_partofspeech_df, jaccard_df = import_nlprocessed_files(UPLOADFILEMANAGE_CSV_PATH=USRUPLOADFILE_DIC_PATH, 
                                                                                                                selected_project_name=selected_project_name, 
                                                                                                                only_raw_df=False)
    sents_array = make_sents_array(questionnaire_raw_df)


    #ページレイアウトの指定
    SPC = StParametersClustering()
    #st.sidebar.header("ページレイアウト")
    #SPC.set_page_columns_rate(side=True)
    noun_length = 25
    adjective_length = 25
    verb_length = 0
    feature_cal_method = "tfidf"
    #特徴量の単語リストの作成
    #word_bag = make_word_bag(word_count_partofspeech_df, 
    #                    noun_length=noun_length, 
    #                    adjective_length=adjective_length, 
    #                    verb_length=verb_length)
    st.subheader(f"{page_number}.2 検索したい単語の指定")
    word_bag = st.multiselect(label="""
    文章中に含まれているか否かを確認したい単語を以下で指定してください。
    初期設定の単語は頻出単語上位10個です。
    検索したい単語が表示されていなかった場合、下の空欄に直接文字を入力して単語を検索・追加してください。""", 
                            options=word_count_partofspeech_df.index, 
                            default=list(word_count_partofspeech_df.index[:100]), 
                            key=int(page_number)
                            )
    #candidates_words_list = make_candidates_words_list(word_count_partofspeech_df, word_length=50)
    #assert "part_of_speech" in word_count_partofspeech_df.columns

    st.subheader(f"{page_number}.3 単語の集計結果")
    st.markdown("""
    各アンケート回答に対してユーザーが指定した単語が含まれている個数を集計した表を表示しています。
    表側に「アンケートの回答」、表頭に「ユーザーが指定した単語」、各セルに「アンケート中の単語の出現数」が表示されています。
    """)
    escape_word_bag = [_.replace("+", "\+") for _ in word_bag]
    word_feature_df = get_count_of_target_pattern(words_in_sent_df["words_in_sent"].values, 
                                            sents_array,
                                            escape_word_bag, #candidates_words_list,
                                            "count")
    word_feature_agg_df = copy.deepcopy(word_feature_df)
    word_feature_agg_df.index = [_[:sentence_length_limit_to_show] if _[:sentence_length_limit_to_show] == _ else f"{_[:sentence_length_limit_to_show]}..." for _ in word_feature_df.index]
    word_feature_agg_df.columns = word_bag
    word_feature_agg_df = word_feature_agg_df.reset_index()
    AgGrid(word_feature_agg_df)
    
    st.subheader(f"{page_number}.4 単語の集計結果のダウンロード")
    st.markdown("集計結果をダウンロードする際は以下をクリックください。")
    download_button_for_dataframe(word_feature_df, 
                                download_file_name=f"{selected_project_name}_単語集計結果.csv")

if __name__ == "__main__":
    main()
