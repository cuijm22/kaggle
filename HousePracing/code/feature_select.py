import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from data_preprocess import load_data, process_data
from data_plot import plot_data
import matplotlib.pyplot as plt
import seaborn as sns


def make_mi_scores(x, y):
    x = x.copy()
    for column in x.select_dtypes(["object", "category"]):
        x[column], _ = x[column].factorize()
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in x.dtypes]
    mi_scores = mutual_info_regression(x, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="mi Scores", index=x.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def drop_uninformative(df, mi_scores):
    return df.loc[:, mi_scores > 0.0]


def label_encode(df):
    x = df.copy()
    for column in x.select_dtypes(["category"]):
        x[column] = x[column].cat.codes
    return x


def apply_pca(x, standardize=True):
    # Standardize
    if standardize:
        x = (x - x.mean(axis=0)) / x.std(axis=0)
    # Create principal components
    pca = PCA()
    x_pca = pca.fit_transform(x)
    #print(str((X_pca.shape))+'!!!!!!!!!!!')
    #print(X_pca[-1])
    # Convert to dataframe
    component_names = [f"PC{i+1}" for i in range(x_pca.shape[1])]
    x_pca = pd.DataFrame(x_pca, index=x.index, columns=component_names)
    # Create loadings
    loadings = pd.DataFrame(
        pca.components_.T,  # transpose the matrix of loadings
        columns=component_names,  # so the columns are the principal components
        index=x.columns,  # and the rows are the original features
    )
    #print(x_pca.shape)
    return pca, x_pca, loadings


def pca_inspired(df):
    x = pd.DataFrame()
    x["Feature1"] = df.GrLivArea + df.TotalBsmtSF
    x["Feature2"] = df.YearRemodAdd * df.TotalBsmtSF
    return x


def pca_components(df, features):
    x = df.loc[:, features]
    _, x_pca, _ = apply_pca(x)
    return x_pca


def create_features(df_train, df_test):
    x = df_train.copy()
    y = x.pop("SalePrice")

    # Mutual Information
    mi_scores = make_mi_scores(x, y)
    #print(mi_scores.shape)
    #print(x.shape)
    x = x.drop(["GarageArea", "1stFlrSF", "TotRmsAbvGrd"], axis=1)
    #print(x.shape)

    x_test = df_test.copy()
    y_test = x_test.pop("SalePrice")
    x_test = x_test.drop(["GarageArea", "1stFlrSF", "TotRmsAbvGrd"], axis=1)
    x = pd.concat([x, x_test])
    y = pd.concat([y, y_test])

    #print(x.shape)
    x = drop_uninformative(x, mi_scores)
    #print(x.shape)

    # PCA
    pca_features = [column for column in x.select_dtypes(["int", "float"])]
    x = x.join(pca_inspired(x))
    #x = x.join(pca_components(X, pca_features))
    #print(type(x))
    #X=X.drop(pca_features,axis=1)
    #print(x.select_dtypes(["int", "float"]).isnull())
    #print(x)

    #print(x.shape)
    x = label_encode(x)
    #print(x.shape)
    x = pd.get_dummies(x)
    #print(x.shape)

    x_test = x.loc[df_test.index, :]
    x.drop(df_test.index, inplace=True)

    #print(x.shape,x_test.shape)
    return x, x_test


if __name__ == '__main__':
    df_train, df_test = load_data()
    df_train, df_test = process_data(df_train, df_test)
    df_train, df_test = create_features(df_train, df_test)
    y_train = df_train.loc[:, "SalePrice"]




