import warnings
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
from IPython.display import display
import pandas as pd
import math
from data_preprocess import load_data

# Set Matplotlib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)

# Mute warnings
warnings.filterwarnings('ignore')


def plot_data(df_train):
    display(df_train.info())

    quantitative = [f for f in df_train.columns if df_train.dtypes[f] != 'object' and str(df_train.dtypes[f]) != 'category']
    quantitative.remove('SalePrice')
    if 'MSSubClass' in quantitative:
        quantitative.remove('MSSubClass')
    qualitative = [f for f in df_train.columns if df_train.dtypes[f] == 'object' or str(df_train.dtypes[f]) == 'category']
    if 'MSSubClass' not in qualitative:
        qualitative.append('MSSubClass')
    print(f"{len(quantitative)} quantitative columns:\n{quantitative}")
    print(f"{len(qualitative)} quantitative columns:\n{qualitative}")

    corr_data(df_train, 'SalePrice')

    missing = df_train.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        missing.sort_values(inplace=True)
        missing.plot.bar()
        plt.show()

    y = df_train['SalePrice']
    describe_column(df_train, 'SalePrice')
    plt.figure(1)
    plt.title('Johnson SU')
    sns.distplot(y, kde=False, fit=stats.johnsonsu)
    plt.figure(2)
    plt.title('Normal')
    sns.distplot(y, kde=False, fit=stats.norm)
    plt.figure(3)
    plt.title('Log Normal')
    sns.distplot(y, kde=False, fit=stats.lognorm)
    plt.figure(4)
    plt.title('Normal After Log Transform')
    sns.distplot(np.log(y), kde=False, fit=stats.norm)
    plt.show()

    test_normality = lambda x: stats.shapiro(x)[1] < 0.01
    normal = pd.DataFrame(df_train[quantitative])
    normal = normal.apply(test_normality)
    print(f"Quantitative features in normal distribution: {not normal.any()}")
    f = pd.melt(df_train, value_vars=quantitative)
    g = sns.FacetGrid(f, col="variable", col_wrap=4, sharex=False, sharey=False)
    g = g.map(sns.distplot, "value")
    plt.show()

    variables = ["SalePrice", "OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath", "YearBuilt"]
    sns.pairplot(df_train[variables])
    plt.show()

    for name in qualitative:
        df_train[name] = df_train[name].astype("category")

    a = anova(df_train, qualitative)
    a['disparity'] = np.log(1. / a['pval'].values)
    sns.barplot(data=a, x='feature', y='disparity')
    x = plt.xticks(rotation=90)
    plt.show()


def describe_column(df, column):
    # descriptive statistics summary
    print(f"{column} info:\n{df[column].describe()}")
    # histogram
    sns.distplot(df[column])
    print(f"{column} Skewness: {df[column].skew()}")
    print(f"{column} Kurtosis: {df[column].kurt()}")


def corr_data(df, column):
    # correlation matrix
    corrmat = df.corr()
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    plt.show()
    # saleprice correlation matrix
    k = 20  # number of variables for heatmap
    cols = corrmat.nlargest(k, column)[column].index
    cm = np.corrcoef(df[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values,
                     xticklabels=cols.values)
    plt.show()


def anova(df, qualitative):
    anv = pd.DataFrame()
    anv['feature'] = qualitative
    pvals = []
    for c in qualitative:
        samples = []
        for cls in df[c].unique():
            s = df[df[c] == cls]['SalePrice'].values
            samples.append(s)
        pval = stats.f_oneway(*samples)[1]
        pvals.append(pval)
    anv['pval'] = pvals
    return anv.sort_values('pval')


if __name__ == '__main__':
    df_train, df_test = load_data()
    plot_data(df_train)

