import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from collections import Counter
from scipy.stats import skew
from scipy.special import boxcox1p

# Mute warnings
warnings.filterwarnings('ignore')


def load_data():
    # load data from file
    data_dir = Path("../../data/")
    df_train = pd.read_csv(data_dir / "train.csv", index_col="Id")
    df_test = pd.read_csv(data_dir / "test.csv", index_col="Id")
    return df_train, df_test


def process_data(df_train, df_test):
    df = pd.concat([df_train, df_test])
    df = encode(df)
    df = fillNA(df)
    df = normalize(df)
    #print(df.loc[:, "SalePrice"][0:5])
    df_train = df.loc[df_train.index, :]
    df_test = df.loc[df_test.index, :]
    df_train = remove_outlier(df_train)
    return df_train, df_test


def encode(df):
    # 类别特征
    features_nom = ["MSSubClass", "MSZoning", "Street", "Alley", "LandContour", "LotConfig", "Neighborhood",
                    "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st",
                    "Exterior2nd", "MasVnrType", "Foundation", "Heating", "CentralAir", "GarageType", "MiscFeature",
                    "SaleType", "SaleCondition"]

    # 类别的级别
    five_levels = ["Po", "Fa", "TA", "Gd", "Ex"]
    ten_levels = list(range(10))

    ordered_levels = {
        "OverallQual": ten_levels,
        "OverallCond": ten_levels,
        "ExterQual": five_levels,
        "ExterCond": five_levels,
        "BsmtQual": five_levels,
        "BsmtCond": five_levels,
        "HeatingQC": five_levels,
        "KitchenQual": five_levels,
        "FireplaceQu": five_levels,
        "GarageQual": five_levels,
        "GarageCond": five_levels,
        "PoolQC": five_levels,
        "LotShape": ["Reg", "IR1", "IR2", "IR3"],
        "LandSlope": ["Sev", "Mod", "Gtl"],
        "BsmtExposure": ["No", "Mn", "Av", "Gd"],
        "BsmtFinType1": ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
        "BsmtFinType2": ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
        "Functional": ["Sal", "Sev", "Maj1", "Maj2", "Mod", "Min2", "Min1", "Typ"],
        "GarageFinish": ["Unf", "RFn", "Fin"],
        "PavedDrive": ["N", "P", "Y"],
        "Utilities": ["ELO", "NoSeWa", "NoSewr", "AllPub"],
        "CentralAir": ["N", "Y"],
        "Electrical": ["Mix", "FuseP", "FuseF", "FuseA", "SBrkr"],
        "Fence": ["MnWw", "GdWo", "MnPrv", "GdPrv"],
    }
    # 转换类别
    for name in features_nom:
        df[name] = df[name].astype("category")
        if "None" not in df[name].cat.categories:
            df[name].cat.add_categories("None", inplace=True)
    # 带级别的类别特征
    for name, levels in ordered_levels.items():
        df[name] = df[name].astype(CategoricalDtype(levels, ordered=True))
        if "None" not in df[name].cat.categories:
            df[name].cat.add_categories("None", inplace=True)
    return df


def fillNA(df):
    for name in df.select_dtypes("number"):
        df[name] = df[name].fillna(0)
    for name in df.select_dtypes("category"):
        df[name] = df[name].fillna("None")
    return df


def outlier_detection_train(df, n, columns):
    rows = []
    will_drop_train = []
    for col in columns:
        Q1 = np.nanpercentile(df[col], 25)
        Q3 = np.nanpercentile(df[col], 75)
        IQR = Q3 - Q1
        outlier_point = 1.5 * IQR
        rows.extend(df[(df[col] < Q1 - outlier_point)|(df[col] > Q3 + outlier_point)].index)
    for r, c in Counter(rows).items():
        if c >= n:
            will_drop_train.append(r)
    return will_drop_train


def normalize(df):
    number_data = [column for column in df.select_dtypes(["int", "float"])]
    number_data.pop()
    #print(numeric_data)
    skewed_vals = df[number_data].apply(lambda x: skew(x)).sort_values()
    for index in skewed_vals.index:
        df[index] = boxcox1p(df[index], 0.15)
    return df


def remove_outlier(df, n=5):
    columns = df.select_dtypes(["float", "int"]).columns
    rows = []
    drop_rows = []
    for col in columns:
        Q1 = np.nanpercentile(df[col], 25)
        Q3 = np.nanpercentile(df[col], 75)
        outlier_val = (Q3 - Q1) * 1.5
        rows.extend(df[(df[col] < Q1 - outlier_val)|(df[col] > Q3 + outlier_val)].index)
    # print(rows)
    # print(f"type of rows element==== {rows[0]}")
    for row, cnt in Counter(rows).items():
        if cnt >= n:
            drop_rows.append(row)
    df.drop(drop_rows, inplace=True, axis=0)
    return df


if __name__ == '__main__':
    load_data()