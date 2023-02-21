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
NODE_CRITERION_PAIRCOUNT_LENGTH_LIMIT = int(os.environ.get("NODE_CRITERION_PAIRCOUNT_LENGTH_LIMIT")) #300

sys.path.append(str(MYREPOPATH / "src"))
from QA_utils import *
from QA_natural_language_processing import ExeMorphologicalAnalysisWithWordconvertdict, ExeNLPPreprocess

@dataclass
class StParametersCooccurenceNW:    
    #反発係数：ノードの位置関係を決める
    coefficient_of_restitution = 0.20
    sequential_color_name_dict = {
        "Reds": px.colors.sequential.Reds_r,
        "Greens": px.colors.sequential.Greens_r,
        "Blues": px.colors.sequential.Blues_r,
        "Oranges": px.colors.sequential.Oranges_r,
        "viridis": px.colors.sequential.Viridis,
        }
    #配色の初期値
    cooccurrence_color_scale_name = sequential_color_name_dict["Reds"]
    #clusteringの配色
    clustering_color = sequential_color_name_dict["Reds"]
    
    # ノードの色の規則
    cooccurrence_node_color_rule_list = ["接続数", "出現数"]
    cooccurrence_node_color_rule = cooccurrence_node_color_rule_list[0]
    
    #cooccurrence nwの描画基準
    node_criterion_jaccardcoef = 0.2
    node_criterion_count = 1
    node_criterion_paircount = 1
    #figureのmargin
    figure_maring_top = 10
    figure_maring_bottom = 10
    figure_maring_left = 10
    figure_maring_right = 10
    #共起ネットワークの配置のシード値
    cooccurrence_randomseed = 0
    #タイトルのフォントサイズを決定
    title_font_size = 20
    #edgeの太さ
    edge_magnification = 10
    #edge（辺）を描画する基準
    cooccurrenceNW_weight_criterion = 0
    # textposition
    nw_textposition_dict = {"上部":'top center', "中央":"middle center", "下部":"bottom center"}
    nw_textposition = nw_textposition_dict["上部"]

    def set_nw_textposition(self, side=False):
        if side:
            self.nw_textposition_name = st.sidebar.selectbox("単語の表示位置",
                                                            self.nw_textposition_dict.keys())                               
        else:
            self.nw_textposition_name = st.selectbox("単語の表示位置",
                                                            self.nw_textposition_dict.keys())
        self.nw_textposition = self.nw_textposition_dict[self.nw_textposition_name]

    
    def set_coefficient_of_restitution(self, side=False):
        coefficient_of_restitution_text = """
        単語間の距離（デフォルト30）:数値を変更すると単語間の距離が変わります。いくつかの数字を試し、見た目を変更してください。
        """
        if side:
            self.coefficient_of_restitution = st.sidebar.number_input(label=coefficient_of_restitution_text,
                                                        min_value=1,
                                                        max_value=100,
                                                        step=5,
                                                        value=30) / 100
        else:
            self.coefficient_of_restitution = st.slider.number_input(label=coefficient_of_restitution_text,
                                                        min_value=1,
                                                        max_value=100,
                                                        step=5,
                                                        value=30) / 100
            
    def set_cooccurrence_colorscalse(self, side=False):
        if side:
            self.cooccurrence_color_scale_name = st.sidebar.selectbox("共起ネットワークの配色",
                                                            self.sequential_color_name_dict.keys())                               
        else:
            self.cooccurrence_color_scale_name = st.selectbox("共起ネットワークの配色",
                                                            self.sequential_color_name_dict.keys())
        self.cooccurrence_color_scale = self.sequential_color_name_dict[self.cooccurrence_color_scale_name]
        
    def set_cooccurrence_node_color_rule(self, side=False):
        if side:
            self.cooccurrence_node_color_rule = st.sidebar.selectbox("ノードの色つけのルール",
                                                                self.cooccurrence_node_color_rule_list)
        else:
            self.cooccurrence_node_color_rule = st.selectbox("ノードの色つけのルール",
                                                                self.cooccurrence_node_color_rule_list)
            
def return_G(jaccard_df, word_count_partofspeech_df, user_multiselected_word_list, StParameters, node_criterion_paircount): 
    user_selected_size = 20 # ユーザーが選んだ文字のノードのサイズ
    selected_size = 10
    display_connection_criteria = 5
    word_count_dict = word_count_partofspeech_df["count"]
    
    assert "first_word_in_pair" in jaccard_df.columns
    assert "second_word_in_pair" in jaccard_df.columns
    assert "jaccard_coef" in jaccard_df.columns
    assert "freq_pair" in jaccard_df.columns
    assert "freq_first_word_in_pair" in jaccard_df.columns
    assert "freq_second_word_in_pair" in jaccard_df.columns
    assert "jaccard_coef" in jaccard_df.columns

    #グラフの作成
    word_connection_dict = {}
    for word_pair, row in jaccard_df.iterrows():
        freq_pair = row["freq_pair"]
        first_word_in_pair = row["first_word_in_pair"]
        second_word_in_pair = row["second_word_in_pair"]
        #freq_first_word_in_pair = row["freq_first_word_in_pair"]
        #freq_second_word_in_pair = row["freq_second_word_in_pair"]
        jaccard_coef = row["jaccard_coef"]
        
        if first_word_in_pair in word_count_dict.keys() and second_word_in_pair in word_count_dict.keys():
            tmp_first_node_weight = {"jaccard_coef": jaccard_coef, "freq_pair": freq_pair, "count":word_count_dict[first_word_in_pair]}
            tmp_second_node_weight = {"jaccard_coef": jaccard_coef, "freq_pair": freq_pair, "count":word_count_dict[second_word_in_pair]}
            
            #firstword
            #node_criterion_paircount: 描画基準の個数=1?        
            if first_word_in_pair in user_multiselected_word_list and tmp_first_node_weight["freq_pair"] > node_criterion_paircount:
                if first_word_in_pair in word_connection_dict.keys():
                    word_connection_dict[first_word_in_pair][second_word_in_pair] =tmp_first_node_weight
                else:
                    word_connection_dict[first_word_in_pair] = {second_word_in_pair: tmp_first_node_weight}
            #secondword
            if second_word_in_pair in user_multiselected_word_list and tmp_second_node_weight["freq_pair"] > node_criterion_paircount:
                if second_word_in_pair in word_connection_dict.keys():
                    word_connection_dict[second_word_in_pair][first_word_in_pair] = tmp_second_node_weight
                else:
                    word_connection_dict[second_word_in_pair] = {first_word_in_pair: tmp_second_node_weight}
                
    #define graph
    graph = nx.Graph()

    ##nodeの設定
    graph.add_nodes_from(user_multiselected_word_list)

    #add edge(辺)
    for user_multiselected_word in user_multiselected_word_list:
        if user_multiselected_word in word_connection_dict.keys():
            difplay_connection_df = pd.DataFrame(word_connection_dict[user_multiselected_word]).T
            difplay_connection_df = difplay_connection_df.sort_values(by="jaccard_coef", ascending=False).head(display_connection_criteria)
            for pair_word, score in difplay_connection_df.iterrows():
                    graph.add_edge(user_multiselected_word, pair_word, weight=score["freq_pair"]) 

    #色の設定
    if StParameters.cooccurrence_node_color_rule == "出現数":
        color_list = [word_count_dict[adjacencies[0]] for node, adjacencies in enumerate(graph.adjacency())]
    elif StParameters.cooccurrence_node_color_rule == "接続数":
        # トレード相手数で色付けもする
        color_list = [len(adjacencies[1]) for node, adjacencies in enumerate(graph.adjacency())]
    else:
        raise ValueError("予期しないノードの着色規則が指定されています。StParameters.cooccurrence_node_color_ruleの値を確認・修正してください。")
    
    #sizeの設定
    size_list = [user_selected_size if adjacencies[0] in user_multiselected_word_list else selected_size for node, adjacencies in enumerate(graph.adjacency())]

    #未接続のノードの削除
    graph.remove_nodes_from(list(nx.isolates(graph)))

    return graph, color_list, size_list       

def make_cooccurrence_nw_plotly(graph, color_list, size_list, StParameters):   
    #変数設定の続き
    #図のmarginの設定
    figure_margin_dict = dict(b=StParameters.figure_maring_bottom,
                    l=StParameters.figure_maring_left,
                    r=StParameters.figure_maring_right,
                    t=StParameters.figure_maring_top)
    figure_annotation = dict(
                        text='text',
                        showarrow=False,
                        xref="paper", 
                        yref="paper",
                        x=0.005, 
                        y=-0.002
                    )

    #positionの指定
    pos = nx.spring_layout(graph, k=StParameters.coefficient_of_restitution, seed=int(StParameters.cooccurrence_randomseed))
    pos_df = pd.DataFrame(pos, index=["x", "y"]).T

    #layoutの決定
    fig = go.Figure(layout=go.Layout(
                    #title='共起ネットワーク',
                    titlefont=dict(size=StParameters.title_font_size, family='Courier New'),
                    showlegend=False,
                    hovermode='closest',
                    margin=figure_margin_dict,
                    annotations=[
                        figure_annotation
                    ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                 )
    )
    #fig.update_layout(
    #    width=StParameters.cooccurrence_width,
    #    height=StParameters.cooccurrence_height,
    #)

    #scatterのtraceを加える
    fig.add_trace(
        go.Scatter(x = pos_df["x"].values,
                   y = pos_df["y"].values,
                   text = pos_df.index,
                   mode = 'markers+text',
                   hoverinfo = 'text',
                   textposition = StParameters.nw_textposition,
                   marker_line_color="black",
                   marker=dict(
                    showscale=True,
                    colorscale=StParameters.cooccurrence_color_scale, #'RdYlBu'
                    reversescale=True,
                    color=color_list,
                    size=size_list, #pos_df.index.map(word_count_dict),#10,
                    #opacity=0.8,
                    colorbar=dict(
                        thickness=15,
                        title=StParameters.cooccurrence_node_color_rule,
                        xanchor='left',
                        titleside='right'
                    ),
                    line=dict(width=2),
                   )
                  ),
    )

    #edgeのtraceを加える
    for edge in graph.edges(data=True):
        # エッジの起点・終点の登録
        x = edge[0]
        y = edge[1]
        weight = edge[2]["weight"]
        xposx, xposy = pos[x]
        yposx, yposy = pos[y]
        
        if weight >= StParameters.cooccurrenceNW_weight_criterion:
            fig.add_trace(
                go.Scatter(x = [xposx, yposx],
                           y = [xposy, yposy],
                           line = dict(width=0.5,color='#888'),
                           opacity=0.5,
                           #line = dict(width=weight * StParameters.edge_magnification, color="#888"),
                           hoverinfo='text',
                               hovertext=str(weight * StParameters.edge_magnification),
                           mode = 'lines+text',
                          ),
            )
    
    return fig

def select_auto_user_multiselected_word_list(word_count_partofspeech_df, 
                                             target_part_of_speech=None, 
                                             word_length=6):
    if target_part_of_speech:
        user_multiselected_word_list = word_count_partofspeech_df[word_count_partofspeech_df["part_of_speech"] == target_part_of_speech].head(word_length).index
    else:
        user_multiselected_word_list = word_count_partofspeech_df.head(word_length).index
    return user_multiselected_word_list

@dataclass
class CooccurenceNWPage:
    def main(self):
        page_number = "6"
        st.header(f"""{page_number}. 共起ネットワーク""")
        st.markdown("""
        共起ネットワークとはアンケート中の **"ペアとして出現"（共起）した単語間** をネットワークとして表現する可視化手法です。\\
        具体的には単語を**点**として表現し、その単語間においてペアとして出現しやすい単語点同士を**線**で繋ぐことで、文章中の単語間の"つながり"の強さをネットワークとして可視化します。\\
        この共起ネットワークによって文章内の単語間の関係性を簡単に把握できます。
        共起ネットワークについて知りたい方は本ページ末に補足説明を追記しているので、確認してください。

        **注意点**
        - このページの操作は実行終了まで時間がかかる可能性があります。ページ右上で「RUNNNING」という文字とともにアイコンが出ている間はアプリが計算を行なっている時間であり、他の操作を行えないことに注意してください。
        - 「RUNNNING」が表示されている限り内部的にはエラーは発生していないため、お待ちいただければいずれ操作可能になります。しかし、特に2分以上「RUNNNING」とアイコンが右上に表示され続け、他の操作が行えない場合は管理者にご連絡ください。

        """)
        st.subheader(f"{page_number}.1 ファイルの選択")
        selected_project_name = select_questionnairefilename_from_management_files(file_manage_path=UPLOADFILEMANAGE_CSV_PATH, 
                                                                                   labeltext="ファイルを選択してください。")

        #ファイルのインポート
        questionnaire_raw_df, words_in_sent_df, word_count_partofspeech_df, jaccard_df = import_nlprocessed_files(UPLOADFILEMANAGE_CSV_PATH=USRUPLOADFILE_DIC_PATH, 
                                                                                                                  selected_project_name=selected_project_name, 
                                                                                                                  only_raw_df=False)

        #st.sidebar.header("ページレイアウト")
        SPCNW = StParametersCooccurenceNW()
        assert "part_of_speech" in word_count_partofspeech_df.columns

        st.sidebar.subheader(f"共起ネットワークのレイアウト設定")
        #SPCNW.set_cooccurrence_randomseed(side=True)
        #SPCNW.set_number_of_words()
        #SPCNW.set_cooccurrenceNW_weight_criterion()
        #SPCNW.set_jaccard_freq_criterion()
        #SPCNW.set_node_criterion_paircount(side=True)
        SPCNW.set_cooccurrence_colorscalse(side=True)
        SPCNW.set_cooccurrence_node_color_rule(side=True)
        SPCNW.set_coefficient_of_restitution(side=True)
        SPCNW.set_nw_textposition(side=True)
            #with st.expander("共起ネットワーク図のレイアウト詳細設定"):
            #SPCNW.set_figure_margin_dict()
            #SPCNW.set_title_font_size()
            #SPCNW.set_edge_magnification() 
        node_criterion_paircount_length_limit = NODE_CRITERION_PAIRCOUNT_LENGTH_LIMIT
    
        if len(questionnaire_raw_df) < node_criterion_paircount_length_limit:
            node_criterion_paircount = 0
        else:
            node_criterion_paircount = 1

        st.subheader(f"{page_number}.2 関係性を知りたい単語の設定")
        st.markdown("""
        本ツールでは、関係性を知りたい単語を指定することで、その単語に中心とした共起ネットワークを作成します。ユーザーは関係性を知りたい単語を選択してください。
        - ユーザーが単語を全て検索・選択する負担を減らすため、ここでは3つの初期単語群（頻出形容詞・頻出単語・頻出名詞）を用意しています。ユーザーは初期単語群を選択後、さらに単語を追加/削除することで、関心のある単語に関する共起ネットワークを作成してください。
        """)
        st.subheader(f"{page_number}.2.1 初期単語群の選択")
        select_word_type = st.selectbox(label=f"初期単語群を選択してください。ここで選択された単語群は{page_number}.2.2の初期単語に反映されます。",
                                        options=["形容詞の中で出現頻度top5の単語", "全単語の中で出現頻度top15の単語", "名詞の中で出現頻度top5の単語"])
        if select_word_type == "全単語の中で出現頻度top15の単語":
            select_word_list = select_auto_user_multiselected_word_list(word_count_partofspeech_df, target_part_of_speech=None, word_length=15)
        elif select_word_type == "名詞の中で出現頻度top5の単語":
            select_word_list = select_auto_user_multiselected_word_list(word_count_partofspeech_df, target_part_of_speech="noun")
        elif select_word_type == "形容詞の中で出現頻度top5の単語":
            select_word_list = select_auto_user_multiselected_word_list(word_count_partofspeech_df, target_part_of_speech="adjective")
        else:
            pass

        st.subheader(f"{page_number}.2.2 単語の選択")
        user_multiselected_word_list = st.multiselect(label=f"""
        関係性を知りたい単語を追加/削除してください。下の欄に文字を入力して単語を検索することもできます。初期設定の単語は{page_number}.2.1 で選択された{select_word_type}です。
        """, 
                                    options=word_count_partofspeech_df.index, 
                                    default=select_word_list.values, 
                                    )

        st.subheader(f"{page_number}.3 共起ネットワーク図")

        if len(user_multiselected_word_list) > 0:
            with st.spinner():
                st.markdown(f"""
                {page_number}.2.2で選ばれた単語が大きな丸点、その単語と共起しやすい単語上位5個が小さな丸点、そして、その単語間の共起しやすさ（"つながり"の強さ）を線の繋がりで表現しています。
                """)
                graph, color_list, size_list = return_G(jaccard_df=jaccard_df, 
                                                        word_count_partofspeech_df=word_count_partofspeech_df, 
                                                        user_multiselected_word_list=select_word_list,
                                                        StParameters=SPCNW,
                                                        node_criterion_paircount=node_criterion_paircount)
                nw_fig = make_cooccurrence_nw_plotly(graph, color_list, size_list, SPCNW)
                st.plotly_chart(nw_fig, use_container_width=False)
        else:
            st.error("単語は少なくとも1つ以上選んでください。")

        st.markdown("""
        **利用上のポイント**
        - **画像をダウンロードしたいとき**：画像をクリックすると画像の右側にアイコン群が表示されます。そのアイコン群の左端のカメラのアイコンをクリックすると画像をダウンロードできます。
        - **画像を拡大したいとき**：画像のある一点をクリックしたまま、（ドラッグアンドドロップの要領で）マウスを動かすとその領域が拡大できます。拡大をやめたいときは画像をダブルクリックしてください。
        **共起ネットワークの補足解説（より詳しく知りたい方向け）**
        - 「ペアとして出現（共起）しやすい」とは **「ある2つの単語において、各々単独での出現数は多くないにも関わらず、どちらかが出現する場合はもう一方も出現する傾向が高いこと」** を指します。例えば、「商品A」と「美味しい」という単語が、単独ではあまり出てこないにも関わらずペアとして出現しやすいなら、「商品A」と「美味しい」は"つながり"が強い（=商品Aは美味しい）と判断します。また、共起ネットワークでは、この単語間に線をつなぐことで、この"つながり"を表現します。
        - 共起ネットワークでは「2単語のペアとしての出現（共起）しやすさ」を「2単語のペアとしての出現回数」に対して「各々単独での出現回数を差し引く」ことで、定量的に評価します。この「各々単独での出現回数を差し引く」ことは、「私」など**単純に日本語として出現しやすい単語が「どの単語ともペアとして出現しやすい単語」と評価されることを避ける**ために導入されています。より具体的にはJaccard係数という指標を採用し、2単語の共起度を評価しています。
        """)

def main():
    st.set_page_config(layout="wide")
    CNWP = CooccurenceNWPage()
    CNWP.main()

if __name__ == "__main__":
    main()