import re
import os
import sys
import MeCab
import random
import shutil
import urllib
import plotly
import datetime
#import neologdn
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
from sklearn.feature_extraction.text import TfidfTransformer

load_dotenv(verbose=True)
REQUIRE_COLUMN_NAME = os.environ.get("REQUIRE_COLUMN_NAME") #sentence

# ストップワードを手に入れる
def get_stopword_list(write_file_path):
    """
    ストップワードを手に入れる。
    """
    if not write_file_path.exists():
        url = 'http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt'
        urllib.request.urlretrieve(url, write_file_path)

    with open(write_file_path, 'r', encoding='utf-8_sig') as file:
        stopword_list = [word.replace('\n', '') for word in file.readlines()]

    return stopword_list


def check_not_None_and_False(target_object):
    if target_object is not None and target_object is not False:
        return True
    else:
        return False

        
def make_sents_array(upload_df):
    return upload_df.loc[:, REQUIRE_COLUMN_NAME].values

def pathlib_glob_without_dotstartfile(target_pathlib_path, 
                                      rglob_or_not=True,
                                      pattern="*"):
    if rglob_or_not:
        file_list = [_ for _ in target_pathlib_path.rglob(pattern) if str(_.stem)[0] != "."]
    else:
        file_list = [_ for _ in target_pathlib_path.glob(pattern) if str(_.stem)[0] != "."]
    return file_list

def set_fileinfo(target_raw_df, fileinfo_save_txt_path, file_name, preprocessing_or_not=False):
    """
    ファイル名, アップロード日,文章数,前処理の有無
    """
    file_info_txt = f"ファイル名,アップロード日,文章数,前処理の有無,アップロード時刻\n{file_name},{str(datetime.date.today())},{str(len(target_raw_df))},{str(preprocessing_or_not)},{str(datetime.datetime.now())}"
    with open(fileinfo_save_txt_path, "w") as f:
        f.write(file_info_txt)

def fillna_df(target_df):
    target_df = target_df.fillna("")
    target_df.index = target_df.index.fillna("")
    return target_df

def read_fileinfo_and_concat(user_file_path):
    """
    各usr_uploadディレクトリからfileinfo.txtファイルを読み込み、DFにまとめて返す。
    
    user_file_path = USRUPLOADFILE_DIC_PATH
    """
    display_columns = ["アップロード日","文章数","前処理の有無"]
    fileinfo_name = "fileinfo"
    uploaddate_col_name = "アップロード日"
    file_col_name = "ファイル名"
    uplaod_date_and_time = "アップロード時刻"

    # 2階層下のfileinfoファイルまで探しにいく
    usr_file_directory_pathes = [_ for _ in user_file_path.rglob("*") if str(_.stem)[0] != "."]
    fileinfo_pathes = [_ for _ in usr_file_directory_pathes if str(_.stem) == fileinfo_name]
    
    # fileinfoファイルを読みに行き、結合し、DF化
    fileinfo_list = []
    for target_fileinfo_path in fileinfo_pathes:
        with open(target_fileinfo_path) as f:
            s = f.read()
            cols = s.split("\n") #カラム名と登録情報を分ける
            assert len(cols) == 2
            fileinfo_list.append(cols[1].split(",")) #登録情報を追加
    fileinfo_df = pd.DataFrame(fileinfo_list, columns=cols[0].split(","))
    # aggridようにインデックスを整理
    fileinfo_df = fileinfo_df.set_index(file_col_name)
    #アップロード時刻でソート
    fileinfo_df = fileinfo_df.sort_values(uplaod_date_and_time)
    #表示するものをコントロール
    fileinfo_df = fileinfo_df.loc[:, display_columns]
            
    return fileinfo_df


def select_questionnairefilename_from_management_files(file_manage_path, labeltext=""):
    """
    file_management_path = UPLOADFILEMANAGE_CSV_PATH

    """
    #管理ファイルの読み込み
    file_management_df = pd.read_csv(file_manage_path, index_col=0)
    selected_project_name = st.selectbox(label=labeltext, 
                                        options=file_management_df.index, 
                                        index=int(len(file_management_df.index)-1))
    return selected_project_name


def import_nlprocessed_files(UPLOADFILEMANAGE_CSV_PATH, selected_project_name, only_raw_df=False):
    """
    プロジェクト名を指定するとファイルをインポートして返す
    """
    selected_project_dic_path = UPLOADFILEMANAGE_CSV_PATH / selected_project_name
    questionnaire_raw_df = pd.read_csv(selected_project_dic_path / "raw_df.csv", index_col=0)
    questionnaire_raw_df = fillna_df(questionnaire_raw_df)
    if only_raw_df:
        return questionnaire_raw_df
    else:
        words_in_sent_df = pd.read_csv(selected_project_dic_path / f"words_in_sent_df.csv", index_col=0)
        word_count_partofspeech_df = pd.read_csv(selected_project_dic_path / f"word_count_partofspeech_df.csv", index_col=0)
        jaccard_df = pd.read_csv(selected_project_dic_path / f"jaccard_df.csv", index_col=0)
        return questionnaire_raw_df, words_in_sent_df, word_count_partofspeech_df, jaccard_df

def get_count_of_target_pattern(target_sents, sents_array, word_bag, count_method="count"):
    """
    sents_array:list
        文章のリスト
    word_bag
        単語のリスト、特徴量となる。
    """
    unusable_symbol_list = ["?", "(", ")"]
    word_feature_dict = {}
    for target_word_pattern in tqdm(word_bag):
        tmp_match_list = []
        for value in target_sents:
            if type(value) == str and target_word_pattern not in unusable_symbol_list:
                counter = re.findall(target_word_pattern, value)
                if count_method == "count" or count_method == "tfidf":
                    tmp_match_list.append(len(counter))
                elif count_method == "bool":
                    tmp_match_list.append(len(counter) > 0)
                else:
                    raise ValueError("count, bool, tfidf以外の計算方法が指定されました。")
            else:
                if count_method == "count" or count_method == "tfidf":
                    tmp_match_list.append(0)
                elif count_method == "bool":
                    tmp_match_list.append(False)
                else:
                    raise ValueError("count, bool, tfidf以外の計算方法が指定されました。")
        word_feature_dict[target_word_pattern] = tmp_match_list
    word_feature_df = pd.DataFrame(word_feature_dict)

    if count_method == "tfidf":
        tfidf = TfidfTransformer(use_idf=True, norm="l2", smooth_idf=True)
        word_feature_df = pd.DataFrame(tfidf.fit_transform(pd.DataFrame(word_feature_dict)).toarray())
    word_feature_df.index=sents_array
    word_feature_df.columns=word_feature_dict.keys()

    return word_feature_df

@dataclass
class StParametersAgGrid:
    #AgGridの配色
    AgGrid_color_map_list = ["blue", "streamlit", "light", "dark", "fresh", "material"]
    AgGrid_color_map = "blue"
    
    def set_AgGrid_color_map(self, side=False):
        if side:
            self.AgGrid_color_map = st.sidebar.selectbox("表の配色を決めてください。",
                                                        self.AgGrid_color_map_list)
        else:
            self.AgGrid_color_map = st.selectbox("表の配色を決めてください。",
                                                self.AgGrid_color_map_list)


def download_button_for_dataframe(target_dataframe, download_file_name, labeltext="ダウンロードボタン", ):
    target_dataframe = target_dataframe.to_csv().encode('utf-8_sig')
    st.download_button(
        label=labeltext,
        data=target_dataframe,
        file_name=download_file_name,
        mime='text/csv',
    )

def shorten_ix_for_Aggrid(target_agg_df, sentence_length_limit_to_show):
    target_agg_df.index = [_[:sentence_length_limit_to_show] if _[:sentence_length_limit_to_show] == _ else f"{_[:sentence_length_limit_to_show]}..." for _ in target_agg_df.index]
    target_agg_df = target_agg_df.reset_index()
    return target_agg_df