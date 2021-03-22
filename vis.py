import pandas as pd
import numpy as np
import pathlib
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

def plot_intersection(left, right, column, set_labels, ax=None):
    left_set = set(left[column])
    right_set = set(right[column])
    venn2(subsets=(left_set, right_set), set_labels=set_labels, ax=ax)
    return ax

def plot_right_left_inersection(train_df, test_df, columns='__all__', output_path: pathlib.Path = None):
    """2つのデータフレームのカラムの共通集合を可視化"""
    if columns == '__all__':
        columns = set(train_df.columns) & set(test_df.columns)

    columns = list(columns)
    nfigs = len(columns)
    ncols = 6
    nrows = - (- nfigs // ncols)
    fig, axes = plt.subplots(figsize=(3 * ncols, 3 * nrows), ncols=ncols, nrows=nrows)
    axes = np.ravel(axes)
    for c, ax in zip(columns, axes):
        plot_intersection(train_df, test_df, column=c, set_labels=('Train', 'Test'), ax=ax)
        ax.set_title(c)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path.joinpath('inersection.png'))
    return fig, ax

def check_null_count(train: pd.DataFrame, test: pd.DataFrame, output_path: pathlib.Path = None):

    output_path.mkdir(parents=True, exist_ok=True)

    null_count: pd.DataFrame = pd.concat([train.isnull().sum(), test.isnull().sum()], axis=1)
    null_count.columns = ["train", "test"]

    print(null_count)

    if output_path is not None:
        null_count.to_csv(output_path.joinpath('null_count.csv'))
        fig = null_count.plot.barh().get_figure()
        fig.tight_layout()
        fig.savefig(output_path.joinpath('null_count.png'))
