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
import plotly.graph_objs as go
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.utils import shuffle
from dataclasses import dataclass
import matplotlib.colors as mcolors
import streamlit.components.v1 as components


@dataclass
class StParametersClustering:
    sequential_color_name_dict = {
        "Reds": px.colors.sequential.Reds_r[1::3],
        "Reds2": px.colors.sequential.Reds_r,
        "Greens": px.colors.sequential.Greens_r[1::3],
        "Greens2": px.colors.sequential.Greens_r,
        "Blues": px.colors.sequential.Blues_r[1::3],
        "Blues2": px.colors.sequential.Blues_r,
        "Oranges": px.colors.sequential.Oranges_r[1::3],
        "Orange2": px.colors.sequential.Oranges_r,
        "Pastel": px.colors.qualitative.Pastel,
        "Pastel1": px.colors.qualitative.Pastel1,
        "Pastel2": px.colors.qualitative.Pastel2,
        "Bold": px.colors.qualitative.Bold,
        "viridis": px.colors.sequential.Viridis[1::3],
        "Set1": px.colors.qualitative.Set1[1::3],
        "Set2": px.colors.qualitative.Set2[1::3],
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


def visualize_pie_chart(match_pattern_df, pie_label, SP):
    """_summary_

    Args:
        match_pattern_df (_type_): _description_
        pie_label (_type_): _description_
        SP (_type_): SPC = StParametersClustering()

    Returns:
        _type_: _description_
    """
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

def visualize_hist(hist_target_list, show_name):
    hist_fig = go.Figure(data=[go.Histogram(x=hist_target_list)])
    hist_fig.update_layout(
                  xaxis=dict(title=show_name))
    return hist_fig
    

@dataclass
class StParametersStackBar:
    sequential_color_name_dict = {
        "Pastel": px.colors.qualitative.Pastel,
        "Pastel1": px.colors.qualitative.Pastel1,
        "Pastel2": px.colors.qualitative.Pastel2,
        "Reds": px.colors.sequential.Reds_r[1:],
        "Greens": px.colors.sequential.Greens_r[1:],
        "Blues": px.colors.sequential.Blues_r[1:],
        "Oranges": px.colors.sequential.Oranges_r[1:],
        "Bold": px.colors.qualitative.Bold,
        "viridis": px.colors.sequential.Viridis,
        "Set1": px.colors.qualitative.Set1,
        "Set2": px.colors.qualitative.Set2,
        }
    # pie chartの色
    stackbar_sequential_color = sequential_color_name_dict["Pastel"]
    #pieのサイズ
    pie_width = 600
    pie_height = 400
            
    def set_stackbar_color(self, side=False):
        if side:
            sequential_color_name = st.sidebar.selectbox("配色を指定してください。",
                                                         self.sequential_color_name_dict.keys())
        else:
            sequential_color_name = st.selectbox("配色を指定してください。",
                                                 self.sequential_color_name_dict.keys())            
        self.stackbar_sequential_color = self.sequential_color_name_dict[sequential_color_name]

    def set_stackbar_fig_size(self, side=False):
        if side:
            self.pie_width = st.sidebar.slider(label="figure width（デフォルト600）",
                                                    min_value=100,
                                                    max_value=1000,
                                                    value=600)
            self.pie_height = st.sidebar.slider(label="figure height（デフォルト400）",
                                                        min_value=100,
                                                        max_value=1000,
                                                        value=400)
        else:
            self.pie_width = st.slider(label="figure width（デフォルト600）",
                                                    min_value=100,
                                                    max_value=1000,
                                                    value=600)
            self.pie_height = st.slider(label="figure height（デフォルト400）",
                                                        min_value=100,
                                                        max_value=1000,
                                                        value=400)