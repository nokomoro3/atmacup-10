import numpy as np
import pandas as pd
import re
import pathlib
import pycld2 as cld2
import pickle
from fasttext import load_model
from sklearn.preprocessing import LabelEncoder
import nltk
import texthero as hero
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from gensim.models import word2vec, KeyedVectors
from PIL import ImageColor
import os
import hashlib
import colorsys
from sklearn.model_selection import StratifiedKFold

os.environ["PYTHONHASHSEED"] = "0"

class AbstractBaseBlock:
    """特徴量作成ブロックのベースクラス"""
    def fit(self, input_df: pd.DataFrame, y=None):
        return self.transform(input_df)

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

class NumericBlock(AbstractBaseBlock):
    """数値そのものを特徴量として扱う"""
    def __init__(self, column):
        self.column = column

    def transform(self, input_df):
        out_df = pd.DataFrame()
        out_df[self.column] = input_df[self.column]
        return out_df

class StringLengthBlock(AbstractBaseBlock):
    """文字列データの文字列長を特徴量として扱う"""
    def __init__(self, column):
        self.column = column

    def transform(self, input_df):
        out_df = pd.DataFrame()
        out_df[self.column] = input_df[self.column].str.len()
        return out_df

class CountEncodingBlock(AbstractBaseBlock):
    """文字列データのデータ全体に対する出現回数でエンコード"""
    def __init__(self, column: str):
        self.column = column

    def fit(self, input_df, y=None):
        vc = input_df[self.column].value_counts()
        self.count_ = vc
        return self.transform(input_df)

    def transform(self, input_df):
        out_df = pd.DataFrame()
        out_df[self.column] = input_df[self.column].map(self.count_)
        return out_df

class LabelEncodingBlock(AbstractBaseBlock):
    """文字列データの種類毎に数値を割り当てるラベルエンコード"""
    def __init__(self, column: str):
        self.column = column

    def fit(self, input_df, y=None):
        self.label_enc = LabelEncoder()
        self.label_enc.fit(input_df[f"{self.column}"])
        
    def transform(self, input_df):
        out_df = pd.DataFrame()
        out_df[self.column] = self.label_enc.transform(input_df[self.column])
        return out_df

class OneHotEncoding(AbstractBaseBlock):
    """文字列データの種類毎にカラムを割り当てるワンホットエンコード"""
    def __init__(self, column, min_count=30):
        self.column = column
        self.min_count = min_count

    def fit(self, input_df, y=None):
        x = input_df[self.column]
        vc = x.value_counts()
        categories = vc[vc > self.min_count].index
        self.categories_ = categories

        return self.transform(input_df)

    def transform(self, input_df):
        x = input_df[self.column]
        cat = pd.Categorical(x, categories=self.categories_)
        out_df = pd.get_dummies(cat)
        out_df.columns = out_df.columns.tolist()
        return out_df.add_prefix(f'{self.column}=')

# class TargetEncodingBlock(AbstractBaseBlock):
#     """文字列データの種類毎に数値を割り当てるラベルエンコード"""
#     def __init__(self, column: str):
#         self.column = column

#     def fit(self, input_df, y=None):

#         cat_features = [self.column]
#         cbe = CatBoostEncoder(cols=cat_features)

#         X = input_df.drop(['likes'],axis=1)
#         y = input_df[['likes']]
        
#         cbe.fit(input_df[cat_features],input_df[['likes']])

#         input_df=input_df.join(cbe.transform(input_df[cat_features]).add_suffix('_target'))
#         input_df.drop(cat_features,axis=1,inplace=True)

#         input_df=input_df.join(cbe.transform(input_df[cat_features]).add_suffix('_target'))
#         input_df.drop(cat_features,axis=1,inplace=True)

#         # train_df = input_df[~(input_df.likes.isnull())]

#         # agg_df = train_df.groupby(self.column).agg({'likes': ['sum', 'count']})

#         # folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

#         # ts = pd.Series(np.empty(input_df.shape[0]), index=input_df.index)       
#         # for _, holdout_idx in folds.split(input_df, input_df.likes):
#         #     # ホールドアウトする行を取り出す
#         #     holdout_df = input_df.iloc[holdout_idx]

#         #     # ホールドアウトしたデータで合計とカウントを計算する
#         #     holdout_agg_df = holdout_df.groupby(self.column).agg({'likes': ['sum', 'count']})

#         #     # 全体の集計からホールドアウトした分を引く
#         #     train_agg_df = agg_df - holdout_agg_df

#         #     # ホールドアウトしたデータの平均値を計算していく
#         #     oof_ts = holdout_df.apply(lambda row: train_agg_df.loc[row.likes][('likes', 'sum')] \
#         #                                           / (train_agg_df.loc[row.likes][('likes', 'count')] + 1), axis=1)
#         #     # 生成した特徴量を記録する
#         #     ts[oof_ts.index] = oof_ts

#         # print(ts)
        
#     def transform(self, input_df):
#         out_df = pd.DataFrame()
#         out_df[self.column] = self.label_enc.transform(input_df[self.column])
#         return out_df

class ObjectSizeBlock(AbstractBaseBlock):
    """オブジェクトのサイズ情報を抽出した特徴量"""
    def transform(self, input_df):

        out_df = pd.DataFrame()

        for axis in ['h', 'w', 't', 'd']:
            column_name = f'size_{axis}'
            size_info = input_df['sub_title'].str.extract(r'{} (\d*|\d*\.\d*)(cm|mm)'.format(axis)) # 正規表現を使ってサイズを抽出
            size_info = size_info.rename(columns={0: column_name, 1: 'unit'})
            size_info[column_name] = size_info[column_name].replace('', np.nan).astype(float) # dtypeがobjectになってるのでfloatに直す
            size_info[column_name] = size_info.apply(lambda row: row[column_name] * 10 if row['unit'] == 'cm' else row[column_name], axis=1) # 　単位をmmに統一する
            out_df[column_name] = size_info[column_name] # trainにくっつける

        out_df = out_df.fillna(0)

        return out_df

class DatingYearBlock(AbstractBaseBlock):
    """制作期間の長さを計算した特徴量"""
    def transform(self, input_df):

        out_df = pd.DataFrame()
        out_df['dating_interval'] = input_df['dating_year_early'] - input_df['dating_year_late']
        out_df = out_df.fillna(0)
        return out_df

class AcquisitionDateNumeric(AbstractBaseBlock):
    """収集日を数値データ化した特徴量"""
    def transform(self, input_df):

        out_df = pd.DataFrame()
        out_df['acquisition_date'] = input_df["acquisition_date"].fillna("0").map(lambda x: int(x[0:4]))
        return out_df

def left_join(left, right, on='object_id'):
    if isinstance(left, pd.DataFrame):
        left = left[on]
    return pd.merge(left, right, on=on, how='left').drop(columns=[on])

class ExternalTableBlock(AbstractBaseBlock):
    """外部テーブルデータの結合による特徴量。"""
    """種類の多さをcount、他にmin_count以上のものをワンホットエンコーディングする。"""
    def __init__(self, info_file, column="name", min_count=30):
        self.min_count = min_count
        self.info_file = info_file
        self.column = column

    def fit(self, input_df, y=None):
        info_df = pd.read_csv(self.info_file)

        vc = info_df[self.column].value_counts()
        # 出現回数 min_count 以上に絞る
        use_names = vc[vc >= self.min_count].index

        # isin で min_count 回以上でてくるようなレコードに絞り込んでから corsstab を行なう
        idx = info_df[self.column].isin(use_names)
        _use_df = info_df[idx].reset_index(drop=True)

        self.agg_df_ = pd.crosstab(_use_df['object_id'], _use_df[self.column])
        self.agg_df_ = self.agg_df_.add_prefix(f'{self.info_file.stem}_{self.column}=')

        # 種類の多さも特徴量として計算
        self.count_df_ = info_df.groupby('object_id').agg(count=(f'{self.column}','count'))
        self.count_df_ = self.count_df_.add_prefix(f'{self.info_file.stem}_')

        return self.transform(input_df)

    def transform(self, input_df):
        out_df = pd.concat([
            left_join(input_df, self.agg_df_),
            left_join(input_df, self.count_df_)
        ], axis=1)
        out_df = out_df.fillna(0.0)
        return out_df

def hashfxn(x):
    return int(hashlib.md5(str(x).encode()).hexdigest(), 16)

class Word2VecExternalTableBlock(AbstractBaseBlock):
    def __init__(self, info_file):
        self.info_file = info_file

    def fit(self, input_df, y=None):

        info_df = pd.read_csv(self.info_file)

        df_group = info_df.groupby("object_id")["name"].apply(list).reset_index()

        vocab = set()
        for words in info_df["name"].unique():
            for w in re.split('[^A-Za-z0-9]+', words):
                vocab.add(w)

        w2v_model = word2vec.Word2Vec(df_group["name"].values.tolist(),
            size=len(vocab), min_count=1, window=1, iter=100, hashfxn=hashfxn)

        sentence_vectors = df_group["name"].apply(
            lambda x: np.mean([w2v_model.wv[e] for e in x], axis=0)
        )
        sentence_vectors = np.vstack([x for x in sentence_vectors])
        self.sentence_vector_df = pd.DataFrame(
            sentence_vectors,
            columns=[f"{self.info_file.stem}_w2v_{i}" for i in range(len(vocab))])
        self.sentence_vector_df.index = df_group["object_id"]

        return self.transform(input_df)

    def transform(self, input_df):
        out_df = pd.concat([
            left_join(input_df, self.sentence_vector_df),
        ], axis=1)
        out_df = out_df.fillna(0.0)
        return out_df


class PaletteStatsBlock(AbstractBaseBlock):
    """Paletteという外部テーブルデータの結合による特徴量。"""
    """色の平均、分散や色相(hue)。RGBだけでなくYCbCrへの変換特徴量も使用。"""
    def __init__(self, info_file, pkl_file=None, reload=False):
        
        self.info_file = info_file
        
        if (pkl_file is not None) and (pkl_file.exists() == True) and (reload == False):
            # load
            with open(pkl_file, mode='rb') as fp:
                self.palette_stats_ = pickle.load(fp)

        else:
            palette = pd.read_csv(self.info_file)

            def calc_stats(d):
                name = d.name

                def rgb_to_ycbcr(r,g,b):
                    y  =  0.299 * r + 0.587 * g + 0.114 * b
                    cb = -0.169 * r - 0.331 * g + 0.500 * b
                    cr =  0.500 * r - 0.419 * g - 0.081 * b
                    return y, cb, cr

                # hls_df = pd.DataFrame([*map(colorsys.rgb_to_hls, d.color_r, d.color_g, d.color_b)], columns=['hls_h','hls_l','hls_s'])
                yiq_df = pd.DataFrame([*map(colorsys.rgb_to_yiq, d.color_r, d.color_g, d.color_b)], columns=['yiq___y_','yiq___i_','yiq___q_'])
                hsv_df = pd.DataFrame([*map(colorsys.rgb_to_hsv, d.color_r, d.color_g, d.color_b)], columns=['hsv___h_','hsv___s_','hsv___v_'])
                ycbcr_df = pd.DataFrame([*map(rgb_to_ycbcr, d.color_r, d.color_g, d.color_b)], columns=['ycbcr_y_','ycbcr_cb','ycbcr_cr'])
                d = d.reset_index(drop=True)
                d = pd.concat([d, yiq_df, hsv_df, ycbcr_df], axis=1)
                
                return pd.DataFrame({
                    'rgb___r__mean_raw': (1.0 * d.color_r ).mean(),
                    'rgb___g__mean_raw': (1.0 * d.color_g ).mean(),
                    'rgb___b__mean_raw': (1.0 * d.color_b ).mean(),
                    'yiq___y__mean_raw': (1.0 * d.yiq___y_).mean(),
                    'yiq___i__mean_raw': (1.0 * d.yiq___i_).mean(),
                    'yiq___q__mean_raw': (1.0 * d.yiq___q_).mean(),
                    'hsv___h__mean_raw': (1.0 * d.hsv___h_).mean(),
                    'hsv___s__mean_raw': (1.0 * d.hsv___s_).mean(),
                    'hsv___v__mean_raw': (1.0 * d.hsv___v_).mean(),
                    'ycbcr_y__mean_raw': (1.0 * d.ycbcr_y_).mean(),
                    'ycbcr_cb_mean_raw': (1.0 * d.ycbcr_cb).mean(),
                    'ycbcr_cr_mean_raw': (1.0 * d.ycbcr_cr).mean(),
                    'rgb___r__mean_raw': (1.0 * d.color_r ).std (),
                    'rgb___g__mean_raw': (1.0 * d.color_g ).std (),
                    'rgb___b__mean_raw': (1.0 * d.color_b ).std (),
                    'yiq___y__mean_raw': (1.0 * d.yiq___y_).std (),
                    'yiq___i__mean_raw': (1.0 * d.yiq___i_).std (),
                    'yiq___q__mean_raw': (1.0 * d.yiq___q_).std (),
                    'hsv___h__mean_raw': (1.0 * d.hsv___h_).std (),
                    'hsv___s__mean_raw': (1.0 * d.hsv___s_).std (),
                    'hsv___v__mean_raw': (1.0 * d.hsv___v_).std (),
                    'ycbcr_y__mean_raw': (1.0 * d.ycbcr_y_).std (),
                    'ycbcr_cb_mean_raw': (1.0 * d.ycbcr_cb).std (),
                    'ycbcr_cr_mean_raw': (1.0 * d.ycbcr_cr).std (),
                    'rgb___r__mima_raw': (1.0 * d.color_r ).max () - (1.0 * d.color_r ).min (),
                    'rgb___g__mima_raw': (1.0 * d.color_g ).max () - (1.0 * d.color_g ).min (),
                    'rgb___b__mima_raw': (1.0 * d.color_b ).max () - (1.0 * d.color_b ).min (),
                    'yiq___y__mima_raw': (1.0 * d.yiq___y_).max () - (1.0 * d.yiq___y_).min (),
                    'yiq___i__mima_raw': (1.0 * d.yiq___i_).max () - (1.0 * d.yiq___i_).min (),
                    'yiq___q__mima_raw': (1.0 * d.yiq___q_).max () - (1.0 * d.yiq___q_).min (),
                    'hsv___h__mima_raw': (1.0 * d.hsv___h_).max () - (1.0 * d.hsv___h_).min (),
                    'hsv___s__mima_raw': (1.0 * d.hsv___s_).max () - (1.0 * d.hsv___s_).min (),
                    'hsv___v__mima_raw': (1.0 * d.hsv___v_).max () - (1.0 * d.hsv___v_).min (),
                    'ycbcr_y__mima_raw': (1.0 * d.ycbcr_y_).max () - (1.0 * d.ycbcr_y_).min (),
                    'ycbcr_cb_mima_raw': (1.0 * d.ycbcr_cb).max () - (1.0 * d.ycbcr_cb).min (),
                    'ycbcr_cr_mima_raw': (1.0 * d.ycbcr_cr).max () - (1.0 * d.ycbcr_cr).min (),
                    'rgb___r__mean_wei': (d.ratio * d.color_r ).mean(),
                    'rgb___g__mean_wei': (d.ratio * d.color_g ).mean(),
                    'rgb___b__mean_wei': (d.ratio * d.color_b ).mean(),
                    'yiq___y__mean_wei': (d.ratio * d.yiq___y_).mean(),
                    'yiq___i__mean_wei': (d.ratio * d.yiq___i_).mean(),
                    'yiq___q__mean_wei': (d.ratio * d.yiq___q_).mean(),
                    'hsv___h__mean_wei': (d.ratio * d.hsv___h_).mean(),
                    'hsv___s__mean_wei': (d.ratio * d.hsv___s_).mean(),
                    'hsv___v__mean_wei': (d.ratio * d.hsv___v_).mean(),
                    'ycbcr_y__mean_wei': (d.ratio * d.ycbcr_y_).mean(),
                    'ycbcr_cb_mean_wei': (d.ratio * d.ycbcr_cb).mean(),
                    'ycbcr_cr_mean_wei': (d.ratio * d.ycbcr_cr).mean(),
                    'rgb___r__mean_wei': (d.ratio * d.color_r ).std (),
                    'rgb___g__mean_wei': (d.ratio * d.color_g ).std (),
                    'rgb___b__mean_wei': (d.ratio * d.color_b ).std (),
                    'yiq___y__mean_wei': (d.ratio * d.yiq___y_).std (),
                    'yiq___i__mean_wei': (d.ratio * d.yiq___i_).std (),
                    'yiq___q__mean_wei': (d.ratio * d.yiq___q_).std (),
                    'hsv___h__mean_wei': (d.ratio * d.hsv___h_).std (),
                    'hsv___s__mean_wei': (d.ratio * d.hsv___s_).std (),
                    'hsv___v__mean_wei': (d.ratio * d.hsv___v_).std (),
                    'ycbcr_y__mean_wei': (d.ratio * d.ycbcr_y_).std (),
                    'ycbcr_cb_mean_wei': (d.ratio * d.ycbcr_cb).std (),
                    'ycbcr_cr_mean_wei': (d.ratio * d.ycbcr_cr).std (),
                    'rgb___r__mima_wei': (d.ratio * d.color_r ).max () - (d.ratio * d.color_r ).min (),
                    'rgb___g__mima_wei': (d.ratio * d.color_g ).max () - (d.ratio * d.color_g ).min (),
                    'rgb___b__mima_wei': (d.ratio * d.color_b ).max () - (d.ratio * d.color_b ).min (),
                    'yiq___y__mima_wei': (d.ratio * d.yiq___y_).max () - (d.ratio * d.yiq___y_).min (),
                    'yiq___i__mima_wei': (d.ratio * d.yiq___i_).max () - (d.ratio * d.yiq___i_).min (),
                    'yiq___q__mima_wei': (d.ratio * d.yiq___q_).max () - (d.ratio * d.yiq___q_).min (),
                    'hsv___h__mima_wei': (d.ratio * d.hsv___h_).max () - (d.ratio * d.hsv___h_).min (),
                    'hsv___s__mima_wei': (d.ratio * d.hsv___s_).max () - (d.ratio * d.hsv___s_).min (),
                    'hsv___v__mima_wei': (d.ratio * d.hsv___v_).max () - (d.ratio * d.hsv___v_).min (),
                    'ycbcr_y__mima_wei': (d.ratio * d.ycbcr_y_).max () - (d.ratio * d.ycbcr_y_).min (),
                    'ycbcr_cb_mima_wei': (d.ratio * d.ycbcr_cb).max () - (d.ratio * d.ycbcr_cb).min (),
                    'ycbcr_cr_mima_wei': (d.ratio * d.ycbcr_cr).max () - (d.ratio * d.ycbcr_cr).min (),
                }, index=[name])

            self.palette_stats_ = palette.groupby('object_id', as_index=False).apply(calc_stats)
            self.palette_stats_ = self.palette_stats_.reset_index(level=1).rename(columns={'level_1': 'object_id'})

            if pkl_file is not None:
                # save
                pkl_file.parent.mkdir(parents=True, exist_ok=True)
                with open(pkl_file, mode='wb') as fp:
                    pickle.dump(self.palette_stats_, fp)
        return

    def transform(self, input_df):
        out_df = left_join(input_df, self.palette_stats_)
        out_df = out_df.fillna(0.0)
        return out_df

class ColorStatsBlock(AbstractBaseBlock):
    """colorという外部テーブルデータの結合による特徴量。"""
    """色の平均、分散や色相(hue)。RGBだけでなくYCbCrへの変換特徴量も使用。"""
    def __init__(self, info_file, pkl_file=None, reload=False):
        
        self.info_file = info_file
        
        if (pkl_file is not None) and (pkl_file.exists() == True) and (reload == False):
            # load
            with open(pkl_file, mode='rb') as fp:
                self.color_stats_ = pickle.load(fp)

        else:
            color = pd.read_csv(self.info_file)

            def calc_stats(d):
                name = d.name
                ratio = d.percentage/100
                d["ratio"] = ratio

                rgb_df = pd.DataFrame(d.hex.str.strip().map(ImageColor.getrgb).values.tolist(), columns=['color_r', 'color_g', 'color_b'])
                d = pd.concat([d, rgb_df], axis=1)

                def rgb_to_ycbcr(r,g,b):
                    y  =  0.299 * r + 0.587 * g + 0.114 * b
                    cb = -0.169 * r - 0.331 * g + 0.500 * b
                    cr =  0.500 * r - 0.419 * g - 0.081 * b
                    return y, cb, cr

                # hls_df = pd.DataFrame([*map(colorsys.rgb_to_hls, d.color_r, d.color_g, d.color_b)], columns=['hls_h','hls_l','hls_s'])
                yiq_df = pd.DataFrame([*map(colorsys.rgb_to_yiq, d.color_r, d.color_g, d.color_b)], columns=['yiq___y_','yiq___i_','yiq___q_'])
                hsv_df = pd.DataFrame([*map(colorsys.rgb_to_hsv, d.color_r, d.color_g, d.color_b)], columns=['hsv___h_','hsv___s_','hsv___v_'])
                ycbcr_df = pd.DataFrame([*map(rgb_to_ycbcr, d.color_r, d.color_g, d.color_b)], columns=['ycbcr_y_','ycbcr_cb','ycbcr_cr'])
                d = d.reset_index(drop=True)
                d = pd.concat([d, yiq_df, hsv_df, ycbcr_df], axis=1)

                return pd.DataFrame({
                    'rgb___r__mean_raw': (1.0 * d.color_r ).mean(),
                    'rgb___g__mean_raw': (1.0 * d.color_g ).mean(),
                    'rgb___b__mean_raw': (1.0 * d.color_b ).mean(),
                    'yiq___y__mean_raw': (1.0 * d.yiq___y_).mean(),
                    'yiq___i__mean_raw': (1.0 * d.yiq___i_).mean(),
                    'yiq___q__mean_raw': (1.0 * d.yiq___q_).mean(),
                    'hsv___h__mean_raw': (1.0 * d.hsv___h_).mean(),
                    'hsv___s__mean_raw': (1.0 * d.hsv___s_).mean(),
                    'hsv___v__mean_raw': (1.0 * d.hsv___v_).mean(),
                    'ycbcr_y__mean_raw': (1.0 * d.ycbcr_y_).mean(),
                    'ycbcr_cb_mean_raw': (1.0 * d.ycbcr_cb).mean(),
                    'ycbcr_cr_mean_raw': (1.0 * d.ycbcr_cr).mean(),
                    'rgb___r__mean_raw': (1.0 * d.color_r ).std (),
                    'rgb___g__mean_raw': (1.0 * d.color_g ).std (),
                    'rgb___b__mean_raw': (1.0 * d.color_b ).std (),
                    'yiq___y__mean_raw': (1.0 * d.yiq___y_).std (),
                    'yiq___i__mean_raw': (1.0 * d.yiq___i_).std (),
                    'yiq___q__mean_raw': (1.0 * d.yiq___q_).std (),
                    'hsv___h__mean_raw': (1.0 * d.hsv___h_).std (),
                    'hsv___s__mean_raw': (1.0 * d.hsv___s_).std (),
                    'hsv___v__mean_raw': (1.0 * d.hsv___v_).std (),
                    'ycbcr_y__mean_raw': (1.0 * d.ycbcr_y_).std (),
                    'ycbcr_cb_mean_raw': (1.0 * d.ycbcr_cb).std (),
                    'ycbcr_cr_mean_raw': (1.0 * d.ycbcr_cr).std (),
                    'rgb___r__mima_raw': (1.0 * d.color_r ).max () - (1.0 * d.color_r ).min (),
                    'rgb___g__mima_raw': (1.0 * d.color_g ).max () - (1.0 * d.color_g ).min (),
                    'rgb___b__mima_raw': (1.0 * d.color_b ).max () - (1.0 * d.color_b ).min (),
                    'yiq___y__mima_raw': (1.0 * d.yiq___y_).max () - (1.0 * d.yiq___y_).min (),
                    'yiq___i__mima_raw': (1.0 * d.yiq___i_).max () - (1.0 * d.yiq___i_).min (),
                    'yiq___q__mima_raw': (1.0 * d.yiq___q_).max () - (1.0 * d.yiq___q_).min (),
                    'hsv___h__mima_raw': (1.0 * d.hsv___h_).max () - (1.0 * d.hsv___h_).min (),
                    'hsv___s__mima_raw': (1.0 * d.hsv___s_).max () - (1.0 * d.hsv___s_).min (),
                    'hsv___v__mima_raw': (1.0 * d.hsv___v_).max () - (1.0 * d.hsv___v_).min (),
                    'ycbcr_y__mima_raw': (1.0 * d.ycbcr_y_).max () - (1.0 * d.ycbcr_y_).min (),
                    'ycbcr_cb_mima_raw': (1.0 * d.ycbcr_cb).max () - (1.0 * d.ycbcr_cb).min (),
                    'ycbcr_cr_mima_raw': (1.0 * d.ycbcr_cr).max () - (1.0 * d.ycbcr_cr).min (),
                    'rgb___r__mean_wei': (d.ratio * d.color_r ).mean(),
                    'rgb___g__mean_wei': (d.ratio * d.color_g ).mean(),
                    'rgb___b__mean_wei': (d.ratio * d.color_b ).mean(),
                    'yiq___y__mean_wei': (d.ratio * d.yiq___y_).mean(),
                    'yiq___i__mean_wei': (d.ratio * d.yiq___i_).mean(),
                    'yiq___q__mean_wei': (d.ratio * d.yiq___q_).mean(),
                    'hsv___h__mean_wei': (d.ratio * d.hsv___h_).mean(),
                    'hsv___s__mean_wei': (d.ratio * d.hsv___s_).mean(),
                    'hsv___v__mean_wei': (d.ratio * d.hsv___v_).mean(),
                    'ycbcr_y__mean_wei': (d.ratio * d.ycbcr_y_).mean(),
                    'ycbcr_cb_mean_wei': (d.ratio * d.ycbcr_cb).mean(),
                    'ycbcr_cr_mean_wei': (d.ratio * d.ycbcr_cr).mean(),
                    'rgb___r__mean_wei': (d.ratio * d.color_r ).std (),
                    'rgb___g__mean_wei': (d.ratio * d.color_g ).std (),
                    'rgb___b__mean_wei': (d.ratio * d.color_b ).std (),
                    'yiq___y__mean_wei': (d.ratio * d.yiq___y_).std (),
                    'yiq___i__mean_wei': (d.ratio * d.yiq___i_).std (),
                    'yiq___q__mean_wei': (d.ratio * d.yiq___q_).std (),
                    'hsv___h__mean_wei': (d.ratio * d.hsv___h_).std (),
                    'hsv___s__mean_wei': (d.ratio * d.hsv___s_).std (),
                    'hsv___v__mean_wei': (d.ratio * d.hsv___v_).std (),
                    'ycbcr_y__mean_wei': (d.ratio * d.ycbcr_y_).std (),
                    'ycbcr_cb_mean_wei': (d.ratio * d.ycbcr_cb).std (),
                    'ycbcr_cr_mean_wei': (d.ratio * d.ycbcr_cr).std (),
                    'rgb___r__mima_wei': (d.ratio * d.color_r ).max () - (d.ratio * d.color_r ).min (),
                    'rgb___g__mima_wei': (d.ratio * d.color_g ).max () - (d.ratio * d.color_g ).min (),
                    'rgb___b__mima_wei': (d.ratio * d.color_b ).max () - (d.ratio * d.color_b ).min (),
                    'yiq___y__mima_wei': (d.ratio * d.yiq___y_).max () - (d.ratio * d.yiq___y_).min (),
                    'yiq___i__mima_wei': (d.ratio * d.yiq___i_).max () - (d.ratio * d.yiq___i_).min (),
                    'yiq___q__mima_wei': (d.ratio * d.yiq___q_).max () - (d.ratio * d.yiq___q_).min (),
                    'hsv___h__mima_wei': (d.ratio * d.hsv___h_).max () - (d.ratio * d.hsv___h_).min (),
                    'hsv___s__mima_wei': (d.ratio * d.hsv___s_).max () - (d.ratio * d.hsv___s_).min (),
                    'hsv___v__mima_wei': (d.ratio * d.hsv___v_).max () - (d.ratio * d.hsv___v_).min (),
                    'ycbcr_y__mima_wei': (d.ratio * d.ycbcr_y_).max () - (d.ratio * d.ycbcr_y_).min (),
                    'ycbcr_cb_mima_wei': (d.ratio * d.ycbcr_cb).max () - (d.ratio * d.ycbcr_cb).min (),
                    'ycbcr_cr_mima_wei': (d.ratio * d.ycbcr_cr).max () - (d.ratio * d.ycbcr_cr).min (),
                }, index=[name])

            self.color_stats_ = color.groupby('object_id', as_index=False).apply(calc_stats)
            self.color_stats_ = self.color_stats_.reset_index(level=1).rename(columns={'level_1': 'object_id'})

            if pkl_file is not None:
                # save
                pkl_file.parent.mkdir(parents=True, exist_ok=True)
                with open(pkl_file, mode='wb') as fp:
                    pickle.dump(self.color_stats_, fp)
        return

    def transform(self, input_df):
        out_df = left_join(input_df, self.color_stats_)
        out_df = out_df.fillna(0.0)
        return out_df

class LanguageIdentificationBlock(AbstractBaseBlock):
    """言語判定(何国語かを判定)特徴量"""
    def __init__(self, column: str, model_path: pathlib.Path):
        self.column = column
        self.model = load_model(str(model_path))

    def fit(self, input_df, y=None):
        out_df = pd.DataFrame()
        out_df[f"{self.column}_cld2"] = input_df[self.column].fillna("").map(lambda x: cld2.detect(x)[2][0][1])
        out_df[f"{self.column}_fasttext"] = input_df[self.column].fillna("").map(lambda x: self.model.predict(x.replace("\n", ""))[0][0])
        
        self.label_enc_cld2 = LabelEncoder()
        self.label_enc_cld2.fit(out_df[f"{self.column}_cld2"])
        self.label_enc_ft = LabelEncoder()
        self.label_enc_ft.fit(out_df[f"{self.column}_fasttext"])
        
        return
        
    def transform(self, input_df):
        out_df = pd.DataFrame()
        out_df[f"{self.column}_cld2"] = input_df[self.column].fillna("").map(lambda x: cld2.detect(x)[2][0][1])
        out_df[f"{self.column}_fasttext"] = input_df[self.column].fillna("").map(lambda x: self.model.predict(x.replace("\n", ""))[0][0])

        out_df[f"{self.column}_cld2"] = self.label_enc_cld2.transform(out_df[f"{self.column}_cld2"])
        out_df[f"{self.column}_fasttext"] = self.label_enc_ft.transform(out_df[f"{self.column}_fasttext"])

        return out_df

class Word2VecBlock(AbstractBaseBlock):
    """"""
    def __init__(self, column: str):
        self.column = column

    def transform(self, input_df):
        out_df = pd.DataFrame()

        sentences_df = input_df[self.column].fillna("").apply(lambda x: re.split("[ \)\(\.\,\:\;]+", x)).reset_index()
        sentences_df["object_id"] = input_df["object_id"]

        vocab = set()
        for words in sentences_df[self.column]:
            for w in words:
                vocab.add(w)
        if len(vocab)>300:
            size=300
        else:
            size=len(vocab)

        w2v_model = word2vec.Word2Vec(sentences_df[self.column].values.tolist(),
            size=size, min_count=1, window=1, iter=100, hashfxn=hashfxn)

        sentence_vectors = sentences_df[self.column].apply(
            lambda x: np.mean([w2v_model.wv[e] for e in x], axis=0)
        )
        sentence_vectors = np.vstack([x for x in sentence_vectors])
        self.sentence_vector_df = pd.DataFrame(
            sentence_vectors,
            columns=[f"{self.column}_w2v_{i}" for i in range(size)])
        
        return pd.concat([out_df, self.sentence_vector_df], axis=1)

def text_normalization(text):

    # 英語とオランダ語を stopword として指定
    custom_stopwords = nltk.corpus.stopwords.words('dutch') + nltk.corpus.stopwords.words('english')

    x = hero.clean(text, pipeline=[
        hero.preprocessing.fillna,
        hero.preprocessing.lowercase,
        hero.preprocessing.remove_digits,
        hero.preprocessing.remove_punctuation,
        hero.preprocessing.remove_diacritics,
        lambda x: hero.preprocessing.remove_stopwords(x, stopwords=custom_stopwords)
    ])

    return x

class TfidfBlock(AbstractBaseBlock):
    """tfidf x SVD による圧縮を行なう block"""
    def __init__(self, column: str):
        """
        args:
            column: str
                変換対象のカラム名
        """
        self.column = column

    def preprocess(self, input_df):
        x = text_normalization(input_df[self.column])
        return x

    def get_master(self, input_df):
        """tdidfを計算するための全体集合を返す. 
        デフォルトでは fit でわたされた dataframe を使うが, もっと別のデータを使うのも考えられる."""
        return input_df

    def fit(self, input_df, y=None):
        master_df = self.get_master(input_df)
        text = self.preprocess(input_df)
        self.pileline_ = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000)),
            ('svd', TruncatedSVD(n_components=50, random_state=1234)),
        ])

        self.pileline_.fit(text)
        return self.transform(input_df)

    def transform(self, input_df):
        text = self.preprocess(input_df)
        z = self.pileline_.transform(text)

        out_df = pd.DataFrame(z)
        return out_df.add_prefix(f'{self.column}_tfidf_')

def run_blocks(train_df, test_df, blocks, y=None):
    train_out_df = pd.DataFrame()
    test_out_df = pd.DataFrame()

    for block in blocks:

        block.fit(pd.concat([train_df, test_df], ignore_index=True))
            
        feats_tr_df = block.transform(train_df)
        assert len(train_df) == len(feats_tr_df), block

        feats_tt_df = block.transform(test_df)
        assert len(test_df) == len(feats_tt_df), block

        name = block.__class__.__name__
        train_out_df = pd.concat([train_out_df, feats_tr_df.add_prefix(f'{name}#')], axis=1)
        test_out_df = pd.concat([test_out_df, feats_tt_df.add_prefix(f'{name}#')], axis=1)

    return train_out_df, test_out_df
