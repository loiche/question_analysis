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
import itertools
import unicodedata
import collections
import numpy as np
import pandas as pd
from tqdm import tqdm
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv(verbose=True)
MYREPOPATH = Path(os.environ.get("MYREPOPATH"))
NODE_CRITERION_PAIRCOUNT_LENGTH_LIMIT = int(os.environ.get("NODE_CRITERION_PAIRCOUNT_LENGTH_LIMIT")) #300

@dataclass
class ExeMorphologicalAnalysisWithWordconvertdict:
    tagger:MeCab.Tagger()
    stopword_list:list
    word_convert_dict:dict
    StParameters: None
    """
    strのsentを入れたらストップワード以外の名詞が入ったリストが返る。
    - ストップワードの単語は無視
    - 平仮名1文字の単語は無視

    !ここをいろいろな関数にすることで共起ネットワークで取得できる語句が多様化する

    parameters:
    -----------
    sent:str
        形態素解析対象の文章

    tagger: MeCab.Tagger('-Ochasen')
        形態素解析用のタガー（タグ付けプログラム）

    stopword_list: list
        ストップワードが入ったリスト

    Returns:
    ----------
    word_list: list
        名詞が入ったリスト
    """
    def main(self, target_sent):
        #表記揺れの統一
        #target_sent = neologdn.normalize(target_sent)
        #if type(target_sent) == str:
        target_sent = unicodedata.normalize("NFKC", target_sent)
        #else:
        #    target_sent = ""

        katakana_p = re.compile('[\u30A1-\u30FF]+')
        hiragana_p = re.compile('[\u3041-\u309F]+')
        
        # 直後が否定形であるかの識別に使う。
        negative_form_flag = False

        word_part_of_speech_dict = {}
        for analysed_word in self.tagger.parse(target_sent).split('\n')[::-1]: #否定形の判別のためリストの下から開けていく。
            part_of_speech_list = analysed_word.split("\t")
            
            #EOS対策, part_of_speech_listのlengthが1より大きいと処理を続ける。
            if len(part_of_speech_list) > 1:
                pass
            else:
                continue

            #表層形\t品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用型,活用形,原形,読み,発音
            surface = part_of_speech_list[2]
            part_of_speech = part_of_speech_list[3]
            yomi = part_of_speech_list[1]
            word_stem = part_of_speech_list[2]

            # 単語の変換を行う
            if self.word_convert_dict and word_stem in set(self.word_convert_dict.keys()):
                for word_before_conversion, word_after_conversion in self.word_convert_dict.items():
                    word_stem = re.sub(word_before_conversion, word_after_conversion, word_stem)

            # 平仮名1文字, ストップワードに含まれた単語は無視
            if surface in self.stopword_list or word_stem in self.stopword_list:
                continue

            # 数字があったら無視
            if self.StParameters.number_remove_flag and re.fullmatch(r'\d+\.*\d*', word_stem):
                continue

            # ひらがながhiragana_remove_criteria_count文字数以下の単語を無視
            if hiragana_p.fullmatch(word_stem) and len(word_stem) <= self.StParameters.hiragana_remove_criteria_count:
                continue
            
            if self.StParameters.adverb_remove_flag and part_of_speech == "副詞":
                continue

            # 抽出
            if "名詞" in part_of_speech:
                word_part_of_speech_dict[word_stem] = ["noun", part_of_speech, yomi, surface]
                negative_form_flag = False
            elif "動詞" in part_of_speech and "助動詞" not in part_of_speech and "非自立" not in part_of_speech:
                if negative_form_flag:
                    word_part_of_speech_dict[f"{word_stem}+ない"] = ["verb", part_of_speech, f"{yomi}+ない", f"{surface}+ない"]
                else:
                    word_part_of_speech_dict[word_stem] = ["verb", part_of_speech, yomi, surface]    
                #negative_form_flagの初期化
                negative_form_flag = False
            elif "形容詞" in part_of_speech:
                if negative_form_flag:
                    word_part_of_speech_dict[f"{word_stem}+ない"] = ["adjective", part_of_speech, f"{yomi}+ない", f"{surface}+ない"]
                else:
                    word_part_of_speech_dict[word_stem] = ["adjective", part_of_speech, yomi, surface]    
                #negative_form_flagの初期化
                negative_form_flag = False
            elif "副詞" in part_of_speech:
                word_part_of_speech_dict[word_stem] = ["adverb", part_of_speech, yomi, surface]
                negative_form_flag = False
            elif "助動詞" in part_of_speech:
                if surface == "ない":
                    negative_form_flag = True
            else:
                negative_form_flag = False
                #pass

        # カタカナのみの単語を文章中に発見したら名詞-一般として抽出する
        if self.StParameters.katakana_extract_flag:
            for target_katakana_word in katakana_p.findall(target_sent):
                if target_katakana_word in word_part_of_speech_dict.keys():
                    pass
                else:
                    word_part_of_speech_dict[target_katakana_word] = ["noun", "名詞-一般", target_katakana_word, target_katakana_word]

        return word_part_of_speech_dict


@dataclass
class ExeNLPPreprocess:
    sents_array: np.array
    
    def convert_whole_word_part_of_speech_dict_to_whole_word_list(self, word_count_dict:dict) -> list:
        """
        wordcloudや共起ネットワーク作成に用いる
        {} -> ["a", "b", "b", "b", "c"]
        """
        whole_word_list = []
        for tmp_word, tmp_part_of_speech in word_count_dict.items():
            whole_word_list.extend([tmp_word] * tmp_part_of_speech)
        return whole_word_list

    def convert_str_list_to_list(self, str_list:str) -> list:
        """
        "['日月', '趣']" -> ['日月', '趣']
        """
        assert len(re.sub("'", "", str_list)) != len(str_list)
        #シングルクォーテーション, []を削除し、", "で分割
        fixed_list = re.sub("'", "", str_list)[1:-1].split(", ")
        return fixed_list

    def count_word_pair_freq(self, word_pair_list, whole_word_list, word_count_dict):
        """
        単語のペアのリストと単語の全出現リスト（重複あり）からjaccard係数を計算し、まとめたDFを作る.
        """
        #単語のペアのリストをstrに変える
        tmp_word_pair_list = [str(sorted(pair_tuple)) for pair_tuple in word_pair_list]

        #単語のペアの出現数をカウント
        word_pair_count_dict = collections.Counter(tmp_word_pair_list)
        word_pair_count_df = pd.DataFrame(word_pair_count_dict, index=["freq_pair"]).T

        #単語ペアの一番目の単語、二番目の単語を取得する
        word_pair_count_df["first_word_in_pair"] = [self.convert_str_list_to_list(ix)[0] for ix in word_pair_count_df.index]
        word_pair_count_df["second_word_in_pair"] = [self.convert_str_list_to_list(ix)[1] for ix in word_pair_count_df.index]

        #一番目の単語と二番目の単語が違うものだけ保持する。
        word_pair_count_df = word_pair_count_df[word_pair_count_df["first_word_in_pair"] != word_pair_count_df["second_word_in_pair"]]

        #単語ペアの一番目の単語、二番目の単語をその文章中での出現数に変換する
        word_pair_count_df["freq_first_word_in_pair"] = word_pair_count_df["first_word_in_pair"].map(word_count_dict)
        word_pair_count_df["freq_second_word_in_pair"] = word_pair_count_df["second_word_in_pair"].map(word_count_dict)

        #jaccard_coefficientの計算
        word_pair_count_df["jaccard_coef"] = word_pair_count_df["freq_pair"]/(word_pair_count_df["freq_first_word_in_pair"] + word_pair_count_df["freq_second_word_in_pair"] - word_pair_count_df["freq_pair"])

        #jaccard_coefficeintでソート
        word_pair_count_df = word_pair_count_df.sort_values(by="freq_pair", ascending=False)

        return word_pair_count_df


    def extract_jaccard_df_freq_pair_top(self, jaccard_df, 
        node_criterion_paircount_length_limit=NODE_CRITERION_PAIRCOUNT_LENGTH_LIMIT):
        """
        可視化の際のデータフレームの操作を行う。
        criterion:ペアとしての最低出現回数
        メモリ対策で10000行までしか単語ペアを保持しない設定？
        """
        if len(self.sents_array) <= node_criterion_paircount_length_limit:
            pass
        else:
            #jaccard_df = jaccard_df.loc[jaccard_df["freq_pair"] > criterion]
            jaccard_df = jaccard_df.head(10000)
        return jaccard_df

    def make_word_count_partofspeech_dict(self, whole_word_part_of_speech_dict, word_count_dict):
        """
        whole_word_part_of_speech_dictから単語の出現頻度と品詞を整理したデータフレームを作成する
        """
        part_of_speech_ix = "part_of_speech"
        count_ix = "count"
        official_part_of_speech_ix = "formal_part_of_speech"
        pronunce_ix = "pronunciation"
        #whole_word_part_of_speech_dictから単語の出現頻度と品詞を整理する。
        word_count_partofspeech_dict = {word:[partofspeech_list[0], word_count_dict[word], partofspeech_list[1], partofspeech_list[2]] for word, partofspeech_list in whole_word_part_of_speech_dict.items()}
        #データフレームにまとめる
        word_count_partofspeech_df = pd.DataFrame(word_count_partofspeech_dict, index=[part_of_speech_ix, count_ix, official_part_of_speech_ix, pronunce_ix]).T
        #一旦 pronounciation を取り除く
        word_count_partofspeech_df = word_count_partofspeech_df.loc[:, [part_of_speech_ix, count_ix, official_part_of_speech_ix]]
        #ソートする
        word_count_partofspeech_df = word_count_partofspeech_df.sort_values(by=[count_ix], ascending=False)
        
        return word_count_partofspeech_df

    def exe_nlp_preprocess(self, morphological_analysis_function):
        """
        #文章列の取得
        sents_array = uploaded_df.iloc[:, 1].values

        exe_morphological_analysis: obtain_noun_from_sent
        """
        #文章中の全単語、単語ペアを抽出
        whole_word_part_of_speech_dict = {}
        word_pair_list = []
        words_in_sent_list = []
        word_count_dict = {}
        
        for target_sent in tqdm(self.sents_array):
            #品詞等に基づく単語の抽出
            word_part_of_speech_dict = morphological_analysis_function.main(target_sent=target_sent)
            
            #word_part_of_speech_dictにマージ
            for tmp_word, tmp_part_of_speech in word_part_of_speech_dict.items():
                if tmp_word in whole_word_part_of_speech_dict.keys():
                    word_count_dict[tmp_word] += 1
                    pass
                else:
                    whole_word_part_of_speech_dict[tmp_word] = tmp_part_of_speech
                    word_count_dict[tmp_word] = 1

        
            #一文中の単語のペアの取り出し
            pair_words = list(itertools.combinations(word_part_of_speech_dict.keys(), 2))
            word_pair_list.extend(pair_words)

            #for tmp_pair_words in itertools.combinations(word_part_of_speech_dict.keys(), 2):
            #    for each_sent in target_sent.split("。"):
            #        if tmp_pair_words[0] in each_sent and tmp_pair_words[1] in each_sent:
            #            #記録
            #            word_pair_list.extend(tmp_pair_words)

            words_in_sent_list.append(list(word_part_of_speech_dict.keys()))

        #sentsとwordsの数が一緒
        assert len(self.sents_array) == len(words_in_sent_list)
        
        #wordcloud用？
        whole_word_list = self.convert_whole_word_part_of_speech_dict_to_whole_word_list(word_count_dict)
        word_count_partofspeech_df = self.make_word_count_partofspeech_dict(whole_word_part_of_speech_dict, word_count_dict)
        
        #jaccard係数を計算
        jaccard_df = self.count_word_pair_freq(word_pair_list, whole_word_list, word_count_dict)

        #表示量を決める
        jaccard_df = self.extract_jaccard_df_freq_pair_top(jaccard_df)

        #文章の中の単語をまとめたDF
        words_in_sent_df = pd.DataFrame([", ".join(_) for _ in words_in_sent_list], columns=["words_in_sent"], index=self.sents_array)

        return words_in_sent_df, word_count_partofspeech_df, jaccard_df
    
def add_pn_to_word_count_partofspeech_df(word_count_partofspeech_df, pn_noun_dict, pn_declinable_dict):
    """word_count_partofspeech_dfの各単語がポジティブかネガティブか付与する。

    Args:
        word_count_partofspeech_df (pd.DataFrame): word_count_partofspeech_df
        pn_noun_dict (dict): 辞書で名詞のpnのペア、
        pn_declinable_dict (dict): 名詞以外の単語のペア
    """
    #形式のチェック
    assert "part_of_speech" in word_count_partofspeech_df.columns
    #検索用のsetの用意
    noun_set = set(pn_noun_dict.keys())
    declinable_set = set(pn_declinable_dict.keys())

    # ポジネガの付与
    pn_judge_list = []
    for tmp_word, tmp_pos in zip(word_count_partofspeech_df.index, word_count_partofspeech_df["part_of_speech"].values):
        if tmp_pos == "noun":
            if tmp_word in noun_set:
                pn_judge_list.append(pn_noun_dict[tmp_word])
            elif "+" in tmp_word:
                stem_and_nai = tmp_word.split("+")
                stem, nai = stem_and_nai[0], stem_and_nai[1]
                if stem in noun_set and nai == "ない":
                    pn_judge_list.append(pn_noun_dict[tmp_word] * (-1))
                else:
                    pn_judge_list.append(0)
            else:
                pn_judge_list.append(0)
        else:
            if tmp_word in declinable_set:
                pn_judge_list.append(pn_declinable_dict[tmp_word])
            elif "+" in tmp_word:
                stem_and_nai = tmp_word.split("+")
                stem, nai = stem_and_nai[0], stem_and_nai[1]
                if stem in declinable_set and nai == "ない":
                    pn_judge_list.append(pn_declinable_dict[stem] * (-1))
                else:
                    pn_judge_list.append(0)
            else:
                pn_judge_list.append(0)
    
    word_count_partofspeech_df["ポジネガ"] = pn_judge_list
    return word_count_partofspeech_df