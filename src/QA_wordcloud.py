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

@dataclass
class StParametersWordcloud:
    wordcloud_max_words = 50
    #wordcloudの文字サイズ
    wordcloud_min_font_size = 4
    wordcloud_max_font_size = 100
    #wordcloudの画像サイズ
    wordcloud_width = 600
    wordcloud_height = 400
    #wordcloudの背景色
    wordcloud_background_color_list = ["white", "black"]
    wordcloud_background_color = wordcloud_background_color_list[0]    
    #wordcloudの文字の配置
    prefer_horizontal_ratio = 1
    #wordcloudの配色
    wordcloud_color_map_list = ["Reds", "Greens", "Blues", "viridis", "Oranges", "Accent", "cividis"]
    #wordcloud_color_map_list = ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r']
    wordcloud_color_map = "Reds"
    
    def set_wordcloud_font_size(self, side=False):
        if side:
            self.wordcloud_min_font_size, self.wordcloud_max_font_size = st.sidebar.slider(label='ワードクラウドの文字サイズの最小値（左）・最大値（右）',
                                     min_value=1,
                                     max_value=200,
                                     value=(4, 100),
                                     )
        else:
            self.wordcloud_min_font_size, self.wordcloud_max_font_size = st.slider(label='ワードクラウドの文字サイズの最小値（左）・最大値（右）',
                                     min_value=1,
                                     max_value=200,
                                     value=(4, 100),
                                     )
    def set_wordcloud_parameters(self, side=False):
        if side:
            self.wordcloud_width = st.sidebar.slider(label="wordcloud width（デフォルト600）",
                                                    min_value=100,
                                                    max_value=1000,
                                                    value=600)
            self.wordcloud_height = st.sidebar.slider(label="wordcloud height（デフォルト400）",
                                                        min_value=100,
                                                        max_value=1000,
                                                        value=400)
        else:
            self.wordcloud_width = st.slider(label="wordcloud width（デフォルト600）",
                                                    min_value=100,
                                                    max_value=1000,
                                                    value=600)
            self.wordcloud_height = st.slider(label="wordcloud height（デフォルト400）",
                                                        min_value=100,
                                                        max_value=1000,
                                                        value=400)
    def set_wordcloud_background_color(self, side=False):
        if side:
            self.wordcloud_background_color = st.sidebar.selectbox("ワードクラウドの背景色を設定してください。",
                                                          self.wordcloud_background_color_list)
        else:
            self.wordcloud_background_color = st.selectbox("ワードクラウドの背景色を設定してください。",
                                                          self.wordcloud_background_color_list)
    def set_prefer_horizontal_ratio(self, side=False):
        if side:
            self.prefer_horizontal_ratio = st.sidebar.slider(label="ワードクラウド中の横書きの出現確率（100で全て横文字になります。）",
                                                        min_value=1,
                                                        max_value=100,
                                                        value=100) /100
        else:
            self.prefer_horizontal_ratio = st.slider(label="ワードクラウド中の横書きの出現確率（100で全て横文字になります。）",
                                                        min_value=1,
                                                        max_value=100,
                                                        value=100) /100
    def set_wordcloud_color_map(self, side=False):
        if side:
            self.wordcloud_color_map = st.sidebar.selectbox("ワードクラウドの配色を決めてください。",
                                                            self.wordcloud_color_map_list)
        else:
            self.wordcloud_color_map = st.selectbox("ワードクラウドの配色を決めてください。",
                                                            self.wordcloud_color_map_list)
    def set_wordcloud_max_words(self, side=False):
        if side:
            self.wordcloud_max_words = st.sidebar.slider(label="ワードクラウド中の単語表示の最大数",
                                                        min_value=1,
                                                        max_value=100,
                                                        value=50)
        else:
            self.wordcloud_max_words = st.slider(label="ワードクラウド中の単語表示の最大数",
                                                        min_value=1,
                                                        max_value=100,
                                                        value=50)


#@st.cache
def make_wordcloud(whole_word_list:list, 
                    font_path:str, 
                    StParametersWordcloud,
                    #tmp_wordcount_dict,
                    array_flag=True) -> np.array:
    """
    ワードクラウドを作成する。

    parameters
    ------------
    whole_word_list: list
        文章中に登場する全単語をまとめたリスト
    font_path: str
        fontが入ったパス
        font_path = "~/Library/Fonts/RictyDiminished-Regular.ttf"

    Returns
    ----------
    wordcloud_array:np.array()
        画像をarrayにしたもの

    wordcloudを可視化したい場合
    plt.imshow(wordcloud)
    とする。
    """
    #def pos_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    #    tmp_max = 0
    #    for key, value in tmp_wordcount_dict.items():
    #        tmp_max = max(value, tmp_max)
#
    #    selected_sequential_color = StParametersWordcloud.wordcloud_color_map
    #    cmap = cm.get_cmap(selected_sequential_color)
    #    if word in tmp_wordcount_dict.keys():
    #        color_index = tmp_wordcount_dict[word] / tmp_max
    #    else:
    #        color_index = 0
    #    rgb = cmap(color_index)
    #    return mcolors.rgb2hex(rgb)

    #単語列のシャッフル
    random.shuffle(whole_word_list)
    wordcloud_text = " ".join(whole_word_list)
    wordcloud = WordCloud(max_font_size=StParametersWordcloud.wordcloud_max_font_size, 
                          min_font_size=StParametersWordcloud.wordcloud_min_font_size,
                          font_path=font_path, 
                          width=StParametersWordcloud.wordcloud_width, 
                          height=StParametersWordcloud.wordcloud_height,
                          background_color=StParametersWordcloud.wordcloud_background_color,
                          prefer_horizontal=StParametersWordcloud.prefer_horizontal_ratio,
                          colormap=StParametersWordcloud.wordcloud_color_map,
                          max_words=StParametersWordcloud.wordcloud_max_words,
                          regexp=r"\S+",
                          #color_func=pos_color_func,
                          #random_state=StParametersWordcloud.wordcloud_randomstate,
                          ).generate(wordcloud_text)
    
    if array_flag:
        #wordcloud.to_file("./tmp_wordcloud.png")
        wordcloud_array = wordcloud.to_array()
        return wordcloud_array
    
    else:
        return wordcloud

def make_wordcloud_nochache(whole_word_list:list, 
                    font_path:str, 
                    StParametersWordcloud,
                    #tmp_wordcount_dict,
                    array_flag=True) -> np.array:
    """
    ワードクラウドを作成する。

    parameters
    ------------
    whole_word_list: list
        文章中に登場する全単語をまとめたリスト
    font_path: str
        fontが入ったパス
        font_path = "~/Library/Fonts/RictyDiminished-Regular.ttf"

    Returns
    ----------
    wordcloud_array:np.array()
        画像をarrayにしたもの

    wordcloudを可視化したい場合
    plt.imshow(wordcloud)
    とする。
    """
    #def pos_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    #    tmp_max = 0
    #    for key, value in tmp_wordcount_dict.items():
    #        tmp_max = max(value, tmp_max)
#
    #    selected_sequential_color = StParametersWordcloud.wordcloud_color_map
    #    cmap = cm.get_cmap(selected_sequential_color)
    #    if word in tmp_wordcount_dict.keys():
    #        color_index = tmp_wordcount_dict[word] / tmp_max
    #    else:
    #        color_index = 0
    #    rgb = cmap(color_index)
    #    return mcolors.rgb2hex(rgb)

    #単語列のシャッフル
    random.shuffle(whole_word_list)
    wordcloud_text = " ".join(whole_word_list)
    wordcloud = WordCloud(max_font_size=StParametersWordcloud.wordcloud_max_font_size, 
                          min_font_size=StParametersWordcloud.wordcloud_min_font_size,
                          font_path=font_path, 
                          width=StParametersWordcloud.wordcloud_width, 
                          height=StParametersWordcloud.wordcloud_height,
                          background_color=StParametersWordcloud.wordcloud_background_color,
                          prefer_horizontal=StParametersWordcloud.prefer_horizontal_ratio,
                          colormap=StParametersWordcloud.wordcloud_color_map,
                          max_words=StParametersWordcloud.wordcloud_max_words,
                          regexp=r"\S+",
                          #color_func=pos_color_func,
                          #random_state=StParametersWordcloud.wordcloud_randomstate,
                          ).generate(wordcloud_text)
    
    if array_flag:
        #wordcloud.to_file("./tmp_wordcloud.png")
        wordcloud_array = wordcloud.to_array()
        return wordcloud_array
    
    else:
        return wordcloud