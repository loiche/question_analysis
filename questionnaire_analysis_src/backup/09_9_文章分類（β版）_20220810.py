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
from QA_utils import get_stopword_list, check_not_None_and_False, make_sents_array, pathlib_glob_without_dotstartfile, set_fileinfo, select_questionnairefilename_from_management_files, import_nlprocessed_files, read_fileinfo_and_concat, get_count_of_target_pattern
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


def download_button_for_dataframe(target_dataframe, download_file_name, labeltext="ダウンロードボタン", ):
    target_dataframe = target_dataframe.to_csv().encode('utf-8_sig')
    st.download_button(
        label=labeltext,
        data=target_dataframe,
        file_name=download_file_name,
        mime='text/csv',
    )

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
    page_number = "9"
    st.set_page_config(layout="wide")
    st.header(f"{page_number}. 文章分類（β版）")
    #表示する文章の長さ
    sentence_length_limit_to_show = 20
    st.markdown("""
        このページではアンケート文章の自動分類（クラスタリング）を実施します。
        クラスタリングは「単語の出現傾向が似た文章」を「同じ内容に言及している文章」と判断し、その文章を同じクラスタに分類します。
        クラスタリングによりアンケートの回答傾向を簡単に把握することができます。
        ユーザーは「文章をいくつのクラスタに分類するか（クラスタ数）」と「クラスタリングにおいて出現傾向を考慮したい単語群」を選択して下さい。\n
        備考
        - 文章が少ない場合はクラスタリングがうまく行えない場合があります。文章数が100以上が望ましく、30以下は結果の解釈が難しい場合があることに注意してください。       
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
    st.subheader(f"{page_number}.2 文章を分類する基準となる単語の選択")
    word_bag = st.multiselect(label="アンケート回答中のトピックに関係すると思われる単語群を複数選択してください。単語は下の欄に文字を入力して検索して入力もできます。", 
                            options=word_count_partofspeech_df.index, 
                            default=list(word_count_partofspeech_df.index[:20]), 
                            )
    #candidates_words_list = make_candidates_words_list(word_count_partofspeech_df, word_length=50)
    #assert "part_of_speech" in word_count_partofspeech_df.columns

    word_feature_df = get_count_of_target_pattern(words_in_sent_df["words_in_sent"].values, 
                                            sents_array,
                                            word_bag, #candidates_words_list,
                                            "count")
                                
    #クラスタ数の指定
    st.subheader(f"{page_number}.3 クラスタ数の選択")
    cluster_num = st.selectbox('文章をいくつのクラスタに分類するか（クラスタ数）を指定してください。',
                        [2, 3, 4, 5, 6, 7, 8, 9, 10],
                        index=1)

    #単語の数がクラスタ数を下回っていない
    if len(word_bag) >= cluster_num:
        #単語の数がクラスタ数を下回っていない
        #特徴量の計算
        word_feature_df = get_count_of_target_pattern(words_in_sent_df["words_in_sent"].values, 
                                                    sents_array,
                                                    word_bag, #candidates_words_list,
                                                    feature_cal_method)

        #単語が一つもマッチしないものを分離
        match_word_feature_df = word_feature_df[word_feature_df.sum(axis=1) >= 1]
        not_match_word_feature_df = word_feature_df[word_feature_df.sum(axis=1) < 1]

        #kmeans
        km = KMeans(n_clusters=cluster_num,
                init="random",
                n_init=10,
                max_iter=300,
                tol=1e-4,
                random_state=0)
        y_km = km.fit_predict(match_word_feature_df)

        # ユーザーが選択した単語を含む文章と含まない文章に分ける
        match_word_feature_df = match_word_feature_df.assign(cluster_class=y_km)
        not_match_word_feature_df = not_match_word_feature_df.assign(cluster_class=cluster_num)
        concat_word_feature_df = pd.concat([match_word_feature_df, not_match_word_feature_df]).loc[word_feature_df.index, :]

        # PCA
        pca = PCA(n_components=2)
        word_feature_pca_np = pca.fit_transform(word_feature_df)
        word_feature_pca_np += np.random.normal(loc=0.0, scale=0.03, size=word_feature_pca_np.shape)
        #クラスタの二次元可視化
        word_feature_for_word_importance_df = get_count_of_target_pattern(words_in_sent_df["words_in_sent"].values,
                                                                        sents_array,
                                                                        word_bag, 
                                                                        "count")
        # クラスタごとの頻出単語のまとめ
        word_importance_df = cal_word_importance(word_feature_for_word_importance_df, concat_word_feature_df["cluster_class"].values, cluster_num+1)
        #クラスタ名の辞書
        cluster_name_dict = make_cluster_name_dict(word_importance_df, concat_word_feature_df["cluster_class"].values, SPC)

        #クラスタリングの図
        cluster_fig = visualize_kmeans_with_pcs(word_feature_pca_np.T[0], 
                                                word_feature_pca_np.T[1], 
                                                sents_array,
                                                concat_word_feature_df["cluster_class"].values,
                                                cluster_name_dict,
                                                SPC)

        #クラスタの円グラフの可視化
        cluster_ix_name = "cluster"
        cluster_df = pd.DataFrame([f"クラスタ{str(_+1)}" for _ in concat_word_feature_df["cluster_class"].values], columns=[cluster_ix_name], index=sents_array)
        cluster_ix_name_rename_dict = {f"クラスタ{str(ii+1)}": jj for ii, jj in cluster_name_dict.items()}
        cluster_df[cluster_ix_name] = cluster_df[cluster_ix_name].map(cluster_ix_name_rename_dict)
        pie_fig = visualize_pie_chart(cluster_df, cluster_ix_name, SPC)

        #単語の重要度を調べる
        st.subheader(f"{page_number}.4 クラスタリングの結果")

        #クラスタリング結果の詳細
        st.subheader("クラスタリング結果の詳細")
        st.markdown("それぞれの文章がどのクラスタに分類されたのかを示しています。")
        cluster_df_aggrid = pd.concat([cluster_df, word_feature_for_word_importance_df], axis=1)
        cluster_df_aggrid.index = [_[:sentence_length_limit_to_show] if _[:sentence_length_limit_to_show] == _ else f"{_[:sentence_length_limit_to_show]}..." for _ in cluster_df.index]
        cluster_df_aggrid = cluster_df_aggrid.reset_index()
        AgGrid(cluster_df_aggrid)
        download_button_for_dataframe(pd.concat([cluster_df, word_feature_for_word_importance_df], axis=1), 
                                    download_file_name=f"{selected_project_name}_クラスタリング結果詳細.csv")

        st.subheader("文章のクラスタへの分類割合を示す円グラフ")
        st.markdown("各クラスタへの文章の分類割合を示しています。")
        st.plotly_chart(pie_fig, use_container_width=False)

        st.subheader("クラスタごとの単語の出現回数の表")
        st.markdown("クラスタごとの単語の出現回数をカウントした結果を示しています。クラスタごとの単語の出現傾向を把握することでクラスタごとの大まかなトピックを把握できます。")

        #クラスタごとの単語の出現割合
        word_importance_df.columns = [cluster_ix_name_rename_dict[_] if _ in cluster_ix_name_rename_dict.keys() else _ for _ in word_importance_df.columns]
        AgGrid(word_importance_df)

        st.subheader("クラスタリング結果の可視化")
        #        より良いクラスタリング結果を得るためには以下を変更してください。
        #- クラスタ数を変更する：同じクラスタに分類されたのに、配置が分裂している場合は本来は違う性質を持った文章群を同じクラスタに入れている可能性があります。クラスタ数を増やすことで、それらを正しく異なるクラスタに分類できるかもしれません。
        st.markdown("""
        クラスタリング結果を2次元にマッピングしています。同一クラスタに属する文章（似た単語構成を持つ文章）がマップ上の近い位置に配置されます。同じクラスタに属する文章群が一箇所に固まり、分裂していない配置になっていることが一つの良いクラスタリング結果の目安です。
        """)
        st.plotly_chart(cluster_fig, use_container_width=False)

    else:
        st.error("「クラスタリングにおいて出現傾向を考慮したい単語」の数は少なくともクラスタ数を下回らないようにしてください。")

if __name__ == "__main__":
    main()
