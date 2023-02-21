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

sys.path.append(str(MYREPOPATH / "src"))
from QA_utils import *
from QA_natural_language_processing import ExeMorphologicalAnalysisWithWordconvertdict, ExeNLPPreprocess
from QA_wordcloud import make_wordcloud, StParametersWordcloud, make_wordcloud_nochache
from QA_visualization import *

def make_candidates_words_list(word_count_partofspeech_df, word_length=50):
    assert "count" in word_count_partofspeech_df.columns
    candidates_words_list = word_count_partofspeech_df.head(word_length)["count"].index
    return candidates_words_list

@dataclass
class CategoryAnalysis:
    def main(self):
        page_number = "8"
        st.title(f"{page_number}. カテゴリごとの単語の集計")
        st.markdown("""
        カテゴリ名・集計したい単語を指定すると、指定した単語のカテゴリごとの出現数を集計した表を作成します。
        """)

        st.subheader(f"{page_number}.1 ファイルの選択")
        # プロジェクトの指定
        selected_project_name = select_questionnairefilename_from_management_files(file_manage_path=UPLOADFILEMANAGE_CSV_PATH, 
                                                                                   labeltext="ファイルを選択してください。")

        #ファイルのインポート
        questionnaire_raw_df, words_in_sent_df, word_count_partofspeech_df, jaccard_df = import_nlprocessed_files(UPLOADFILEMANAGE_CSV_PATH=USRUPLOADFILE_DIC_PATH, 
                                                                                                                  selected_project_name=selected_project_name, 
                                                                                                                  only_raw_df=False)
        # naを埋める
        questionnaire_raw_df = fillna_df(questionnaire_raw_df)
        words_in_sent_df = fillna_df(words_in_sent_df)
        sents_array = make_sents_array(questionnaire_raw_df)

        # 定数の指定
        feature_cal_method = "count"
        ranking_length = 30
        words_in_sent_col = "words_in_sent"
        limit_level_num = 51
        total_number_by_level_name = "カテゴリごとの文章数"
        
        if len(sents_array) < 200:
            candidate_word_length = len(word_count_partofspeech_df)
        else:
            candidate_word_length = 300

        if len(questionnaire_raw_df.columns[1:]) == 0:
            st.error("アップロードしたファイルにカテゴリの登録がありません。")
        else:
            st.subheader(f"{page_number}.2 カテゴリ名の指定")
            target_user_category_name = st.selectbox("カテゴリ名の選択", questionnaire_raw_df.columns[1:])
            st.subheader(f"{page_number}.3 検索したい単語の指定")
            word_bag = st.multiselect(label="文章中に含まれているか確認したい単語を以下で指定してください。検索したい単語が表示されていなかった場合、下の空欄に直接文字を入力して単語を検索・追加してください。", 
                                    options=word_count_partofspeech_df.index, 
                                    default=list(word_count_partofspeech_df.index[:50]), 
                                    key=int(page_number)
                                    )

            if len(questionnaire_raw_df.loc[:, target_user_category_name].unique()) < limit_level_num:
                st.subheader(f"{page_number}.4 カテゴリごとの集計結果")
                st.markdown(f"""
                ユーザーが指定した単語の各カテゴリごとの出現数を表示しています。 表側に「各カテゴリ」、表頭に「ユーザーが指定した単語」、各セルに「カテゴリごとの単語の出現数」が表示されています。
                表頭の「{total_number_by_level_name}」はカテゴリごとのアンケートの文章数を示しています。
                """)
                word_table_groupby_level_df = {}

                with st.spinner("集計しています。"):
                    ##表の表示
                    #AgGrid(target_category_freqwordname_count_df)
                    for target_level in questionnaire_raw_df[target_user_category_name].unique():
                        # 処理
                        target_level_questionnaire_df = questionnaire_raw_df.groupby(target_user_category_name).get_group(target_level)
                        target_level_words_in_sent_df = words_in_sent_df.iloc[target_level_questionnaire_df.index, :]
                        target_level_sents_array = make_sents_array(target_level_questionnaire_df)

                        # 単語・単語ペアのカウント
                        target_level_word_count_dict = {}
                        target_level_word_pair_list = []
                        for _ in target_level_words_in_sent_df["words_in_sent"].values:
                            for tmp_word in _.split(", "):
                                if tmp_word in target_level_word_count_dict.keys():
                                    target_level_word_count_dict[tmp_word] += 1
                                else:
                                    target_level_word_count_dict[tmp_word] = 1

                            target_level_pair_words = list(itertools.combinations(_.split(", "), 2))
                            target_level_word_pair_list.extend(target_level_pair_words)

                        word_bag_count_per_level_dict = {}
                        # ユーザが選択した単語を集計
                        for selected_word in word_bag:
                            # 水準内にユーザーが選択した単語があれば出現数を記録
                            if selected_word in target_level_word_count_dict.keys():
                                word_bag_count_per_level_dict[selected_word] = target_level_word_count_dict[selected_word]
                            # 水準内にユーザーが選択した単語がなければ0を与える。
                            else:
                                word_bag_count_per_level_dict[selected_word] = 0
                        # 水準ごとに記録
                        word_table_groupby_level_df[target_level] = word_bag_count_per_level_dict
                    # DF化
                    word_table_groupby_level_df = pd.DataFrame(word_table_groupby_level_df).T
                    # 水準ごとのカテゴリ数のカウント
                    sent_count_by_level_df = pd.DataFrame(questionnaire_raw_df[target_user_category_name].value_counts())
                    sent_count_by_level_df.columns = [total_number_by_level_name]
                    # 結合
                    word_table_groupby_level_df = pd.concat([sent_count_by_level_df, word_table_groupby_level_df], axis=1)
                    AgGrid(word_table_groupby_level_df.reset_index())


                    

                    st.subheader(f"{page_number}.5 単語の積み上げグラフ")
                    st.markdown("""カテゴリごとの単語の出現数を積み上げ棒グラフで表す。""")
                    target_df = word_table_groupby_level_df.iloc[:, 1:]
                    SPSB = StParametersStackBar()
                    SPSB.set_stackbar_color()
                    fig = go.Figure()
                    for ii, (tmp_word, tmp_count) in enumerate(target_df.iterrows()):
                        fig.add_trace(go.Bar(x=target_df.columns, y=tmp_count.values, name=tmp_word, marker_color=SPSB.stackbar_sequential_color[ii % len(SPSB.stackbar_sequential_color)]))
                        fig.update_layout(
                            width=1200,
                            height=600,
                        )
                    fig.update_layout(barmode='stack', xaxis={'categoryorder':'total descending'},)
                    st.plotly_chart(fig)

                    st.subheader(f"{page_number}.6 カテゴリごとの集計結果のダウンロード")
                    st.markdown("結果をダウンロードする際は以下をクリックください。")
                    download_button_for_dataframe(word_table_groupby_level_df, 
                                                download_file_name=f"{selected_project_name}_単語集計結果.csv")
                                        
            else:
                st.error(f"カテゴリ内の水準数が多すぎます（{str(limit_level_num)}以上）。アプリが止まる可能性があるため処理を停止しました。")
                #large_level_error_message = """
                #カテゴリを以下に列挙します。
                #"""
                #for num, tmp_level in enumerate(questionnaire_raw_df.loc[:, target_user_category_name].unique()):
                #    large_level_error_message += f"""
                #    {str(num)}. {tmp_level}
                #    """
                #st.markdown(large_level_error_message)



def main():
    st.set_page_config(layout="wide")
    CA = CategoryAnalysis()
    CA.main()

if __name__ == "__main__":
    main()