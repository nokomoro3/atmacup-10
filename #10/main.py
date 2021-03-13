import pandas as pd
import numpy as np
import random
import os
import re
import json
from pandas.core.arrays import categorical
from tqdm import tqdm_notebook as tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.metrics import mean_squared_error
import pathlib
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from pandas_profiling import ProfileReport # profile report を作る用

import feats
import trainers
import vis
import tuning

def calc_loss(y_true, y_pred):
    return  np.sqrt(mean_squared_error(y_true, y_pred))

def main(input_path, output_path):
    train_path = input_path.joinpath("train.csv")
    test_path  = input_path.joinpath("test.csv")
    sub_path   = input_path.joinpath("atmacup10__sample_submission.csv")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    # train_reader = pd.read_csv(train_path, chunksize=500)
    # test_reader = pd.read_csv(test_path, chunksize=500)
    # train: pd.DataFrame = train_reader.get_chunk(500)
    # test: pd.DataFrame = test_reader.get_chunk(500)

    #--------------------------------
    # check unique category
    #--------------------------------
    all_df = pd.concat([train, test], ignore_index=True)
    for c in [
        'principal_or_first_maker',
        'principal_maker',
        'copyright_holder',
        'acquisition_method',
        'acquisition_credit_line',
    ]:
        with open(output_path.joinpath(f'unique_{c}.json'), 'w') as f:
            json.dump(sorted([str(n) for n in all_df[f"{c}"].unique()]), f, indent=4)

    principal_maker = pd.read_csv(input_path.joinpath("principal_maker.csv"))
    with open(output_path.joinpath(f'unique_principal_maker_extable.json'), 'w') as f:
        json.dump(sorted([str(n) for n in principal_maker["maker_name"].unique()]), f, indent=4)

    # #--------------------------------
    # # visualize
    # #--------------------------------
    # vis.check_null_count(train, test, output_path)

    # fig_features = ['object_id', 'art_series_id', 'title', 'description', 'long_title',
    #                 'principal_maker', 'principal_or_first_maker', 'sub_title', 
    #                 'copyright_holder', 'more_title', 'acquisition_method',
    #                 'acquisition_date', 'acquisition_credit_line',
    #                 'dating_presenting_date']
    # vis.plot_right_left_inersection(train, test, columns=fig_features, output_path=output_path)

    # # 動かなかった...
    # report = ProfileReport(train)
    # report.to_file(output_path.joinpath("'train_report.html'"))

    #--------------------------------
    # train
    #--------------------------------
    y = np.log1p(train["likes"])

    #--------------------------------
    # preprocess category
    #--------------------------------
    for i in [
        'principal_or_first_maker',
        'principal_maker',
        'copyright_holder',
        'acquisition_method',
        'acquisition_credit_line',
    ]:
        # お互いに存在しないものをunknownにする
        train.loc[~train[c].isin(test[c].unique()),c] = "unknown"
        test.loc[~test[c].isin(train[c].unique()),c] = "unknown"

        # 不定値もunknownにする
        train.loc[train[c].isnull(),c] = "unknown"
        test.loc[test[c].isnull(),c] = "unknown"

        # anonymousもunknownに統一
        train.loc[train[c]=="anonymous",c] = "unknown"
        test.loc[test[c]=="anonymous",c] = "unknown"

    feature_blocks = [
        *[feats.NumericBlock(c) for c in [
            'dating_sorting_date', 'dating_period', 'dating_year_early',
            'dating_year_late',
        ]],
        *[feats.LabelEncodingBlock(c) for c in [
            'title', 'description', 'long_title',
            'principal_maker', 'principal_or_first_maker', 'sub_title',
            'copyright_holder', 'more_title', 'acquisition_method',
            'acquisition_date', 'acquisition_credit_line', 'dating_presenting_date',
            'dating_sorting_date', 'dating_period', 'dating_year_early',
            'dating_year_late',
        ]],
        # *[feats.OneHotEncoding(c) for c in [
        #     'title', 'description', 'long_title',
        #     'principal_maker', 'principal_or_first_maker', 'sub_title',
        #     'copyright_holder', 'more_title', 'acquisition_method',
        #     'acquisition_date', 'acquisition_credit_line', 'dating_presenting_date',
        #     'dating_sorting_date', 'dating_period', 'dating_year_early',
        #     'dating_year_late',
        # ]],
        *[feats.CountEncodingBlock(c) for c in [
            'art_series_id', 
            'title', 'description', 'long_title',
            'principal_maker', 'principal_or_first_maker', 'sub_title',
            'copyright_holder', 'more_title', 'acquisition_method',
            'acquisition_date', 'acquisition_credit_line', 'dating_presenting_date',
            'dating_sorting_date', 'dating_period', 'dating_year_early',
            'dating_year_late',
        ]],
        *[feats.StringLengthBlock(c) for c in [
            'title', 'description', 'long_title',
            'principal_maker', 'principal_or_first_maker', 'sub_title',
            'copyright_holder', 'more_title', 'acquisition_credit_line',
        ]],

        feats.ObjectSizeBlock(),
        feats.DatingYearBlock(),
        feats.AcquisitionDateNumeric(),

        # 別テーブル情報
        *[feats.ExternalTableBlock(p,c) for p,c in [
            (input_path.joinpath("historical_person.csv"), "name"),
            (input_path.joinpath("material.csv"), "name"),
            (input_path.joinpath("object_collection.csv"), "name"),
            (input_path.joinpath("technique.csv"), "name"),
            (input_path.joinpath("production_place.csv"), "name"),
            (input_path.joinpath("principal_maker.csv"), "maker_name"),
        ]],

        feats.Word2VecExternalTableBlock(input_path.joinpath("material.csv")),
        feats.Word2VecExternalTableBlock(input_path.joinpath("object_collection.csv")),
        feats.Word2VecExternalTableBlock(input_path.joinpath("technique.csv")),

        # *[feats.Word2VecBlock(c) for c in [
        #     'title', 'description', 'long_title',
        #     'principal_maker', 'principal_or_first_maker', 'sub_title',
        #     'copyright_holder', 'more_title', 'acquisition_credit_line',
        # ]],

        # palette統計情報
        feats.PaletteStatsBlock(input_path.joinpath("palette.csv"), pathlib.Path(".").joinpath("palette_stats.pkl"), reload=True),

        # color統計情報
        feats.ColorStatsBlock(input_path.joinpath("color.csv"), pathlib.Path(".").joinpath("color_stats.pkl"), reload=True),

        # 言語判定結果
        *[feats.LanguageIdentificationBlock(c, pathlib.Path(".").joinpath("lid.176.bin")) for c in [
            'title', 'description', 'long_title', 'more_title', 'acquisition_credit_line', 'copyright_holder', 
            'principal_maker', 'principal_or_first_maker',
        ]],

        # TFIDF
        *[feats.TfidfBlock(c) for c in [
            'title', 'description', 'long_title', 'more_title','acquisition_credit_line', 'copyright_holder', 
            'principal_maker', 'principal_or_first_maker',
        ]],
    ]

    train_feat_df, test_feat_df = feats.run_blocks(train, test, blocks=feature_blocks)

    train_feat_df = train_feat_df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_=@ #]+', '', x))
    test_feat_df = test_feat_df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_=@ #]+', '', x))

    # 一時保存
    train_feat_df.to_csv(output_path.joinpath("tran_feat.csv"))
    test_feat_df.to_csv(output_path.joinpath("test_feat.csv"))

    # 読み出し
    train_feat_df = pd.read_csv(output_path.joinpath("tran_feat.csv"), index_col=0)
    test_feat_df = pd.read_csv(output_path.joinpath("test_feat.csv"), index_col=0)

    feat_name = [*train_feat_df.columns]
    categorical_features = []

    with open(output_path.joinpath("feature_names.json"), "wt") as f:
        json.dump(sorted(feat_name), f, indent=4)

    # デフォルトパラメータ
    params = {
        'num_leaves': 32,
        'min_data_in_leaf': 64,
        'objective': 'rmse',
        'max_depth': -1,
        'learning_rate': 0.05,
        "boosting": "gbdt",
        "bagging_freq": 1,
        "bagging_fraction": 0.8,
        "bagging_seed": 0,
        "verbosity": -1,
        'reg_alpha': 0.1,
        'reg_lambda': 0.3,
        'colsample_bytree': 0.7,
        'metric':"rmse",
        'num_threads':6,
    }

    # optunaでチューニングする場合
    # params = tuning.run_optuna(train_feat_df, y, test_feat_df)
    # with open(output_path.joinpath("model_params.json"), "wt") as f:
    #     json.dump(params, f, indent=4)

    score, y_test = tuning.cross_validation(train_feat_df, y, test_feat_df, params)

    y_test_sub = np.expm1(y_test)
    sub = pd.read_csv(sub_path)
    sub["likes"] = y_test_sub

    # 0以下はありえないためクリップ
    sub.loc[sub.likes <= 0,"likes"] = 0
    sub.to_csv(output_path.joinpath("submission.csv"), index=False)

if __name__ == "__main__":
    main(input_path=pathlib.Path("./data"), output_path=pathlib.Path("./outdata"))
