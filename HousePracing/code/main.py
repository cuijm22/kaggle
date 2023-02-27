from feature_select import create_features
from data_preprocess import load_data, process_data
from data_plot import plot_data
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Add by LIHT22
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
import math
from Model_stacking import Model_stacking
from Tune_parameters import Tune_parameters
from sklearn.linear_model import Lasso, Ridge, ElasticNet, BayesianRidge
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

## 均方误差 估计值与真值 偏差
def get_mse(records_real, records_predict):
    if len(records_real) == len(records_predict):
        return sum([(x - y) ** 2 for x, y in zip(records_real, records_predict)]) / len

#(records_real)
    else:
        return None
# 均方根误差
def get_rmse(records_real, records_predict):
    ## 均方根误差(均方误差的算术平方根)
    mse = get_mse(records_real, records_predict)
    if mse:
        return math.sqrt(mse)
    else:
        return None
# 交叉验证以及评估函数
def rmse_cv(model,X,y):
    ## 各折数据集的测试结果的均方根误差
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", 

cv=5))   #cv为数据划分的KFold折数
    return rmse

def main():
    # load and process data
    print("start")
    df_train, df_test = load_data()

    # plot data
    # plot_data(df_train)

    # process data
    df_train, df_test = process_data(df_train, df_test)

    print(df_train.shape)
    print(df_test.shape)
    print(df_train[0:1])
    print('process_data end!')

    # create features
    x_train, x_test = create_features(df_train, df_test)
    y_train = df_train.loc[:, "SalePrice"]
    y_test = df_test.loc[:, "SalePrice"]
    log_y_train = np.log(y_train)
    log_y_test = np.log(y_test)

    print(x_train.shape)
    print(x_test.shape)
    print(x_train[0:1])
    print('create_features end!')
    
    # Obtain the optimal parameters
    '''
    ## 1. Lasso
    print("Best hyperparameter for Lasso:")
    param_grid = {'alpha': [0.01, 0.001, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 

0.0009], 'max_iter': [30000],
                  'random_state': [1]}
    Tune_parameters(Lasso()).grid_get(x_train, log_y_train, param_grid)

    ## 2. ElasticNet
    print("Best hyperparameter for ElasticNet:")
    param_grid = {'alpha': [0.1, 0.2, 0.3, 0.01, 0.02, 0.001, 0.002], 'l1_ratio': [0.1, 

0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                  'max_iter': [10000]}
    Tune_parameters(ElasticNet()).grid_get(x_train, log_y_train, param_grid)
    
    ## 3. BayesianRidge
    print("Best hyperparameter for BayesianRidge:")
    param_grid = {'tol':[0.1, 0.2, 0.3, 0.01, 0.02, 0.001, 0.002],\
                  'alpha_1': [0.01, 0.001, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 

0.0009],\
                  'lambda_1':[0.01, 0.001, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 

0.0009], \
                  'n_iter': [10000]}
    Tune_parameters(BayesianRidge()).grid_get(x_train, log_y_train, param_grid)
    
    ## 4. Ridge
    print("Best hyperparameter for Ridge:")
    param_grid = {'alpha': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]}
    Tune_parameters(Ridge()).grid_get(x_train, log_y_train, param_grid)
   
    ## 5. SVR
    print("Best hyperparameter for SVR:")
    param_grid = {'C': [11, 12, 13, 14, 15, 16, 17, 18], 'kernel': ["rbf"],
                  "gamma": [0.001, 0.0001, 0.0002, 0.0003, 0.0004], "epsilon": [0.008, 

0.009, 0.001, 0.01]}
    Tune_parameters(SVR()).grid_get(x_train, np.ravel(log_y_train), param_grid)

    ## 6. KernelRidge
    print("Best hyperparameter for KernelRidge:")
    param_grid = {'alpha': [0.001, 0.01, 0.1, 0.2, 0.3], 'kernel': ["polynomial"], 

'degree': [1, 2, 3],
                  'coef0': [0.8, 1, 1.2, 1.4, 1.6, 1.8]}
    Tune_parameters(KernelRidge()).grid_get(x_train, log_y_train, param_grid)
    
    ## 7. LGBM
    print("Best hyperparameter for LGBM:")
    param_grid = {
        'max_depth':[4,6,8],
        'num_leaves': [5, 10, 15, 20, 25, 30, 35],
        'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
        'feature_fraction': [0.6, 0.7, 0.8, 0.9, 0.95],
        'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 0.95],
        'bagging_freq': [2, 4, 5, 6, 8],
        'lambda_l1': [0, 0.1, 0.4, 0.5, 0.6],
        'lambda_l2': [0, 10, 15, 35, 40],
        'cat_smooth': [1, 10, 15, 20, 35]
    }
    Tune_parameters(LGBMRegressor()).grid_get(x_train, log_y_train, param_grid)
   
    ## 8. GBR
    GBR = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, 

max_features='sqrt',
                                    min_samples_leaf=15, min_samples_split=10, 

loss='huber', random_state=5)
    GBR.fit(x_train, log_y_train)
    #y_pred_GBR = np.expm1(GBR.predict(x_train))
    print(rmse_cv(GBR, x_train, log_y_train).mean())  
    
    ## 9. XGBoost
    # predication = predict(x_train, log_y_train, X_test) # get optimal hyperparameter 

of xgboost
    xgb = XGBRegressor(max_depth=4, learning_rate=0.01587343857928418, 

n_estimators=2170, min_child_weight=4, \
                       colsample_bytree=0.5738706614196447, 

subsample=0.4057484691334741,\
                       reg_alpha=0.00040928611747651574, \
                       reg_lambda=0.022464435049338722)
    '''

    ### Stacking up all the models above, optimized using xgboost
    # defining models
    lasso = Lasso(alpha=0.0004, random_state=1, max_iter=30000)
    ridge = Ridge(alpha=5)
    svr = SVR(C=18, gamma=0.0001, kernel='rbf', epsilon=0.01)
    ker = KernelRidge(alpha=0.001, kernel='polynomial', degree=1, coef0=1.8)
    ela = ElasticNet(alpha=0.001, l1_ratio=0.1, random_state=3, max_iter=10000)
    bay = BayesianRidge(alpha_1=0.01,lambda_1=0.0004,n_iter=10000, tol=0.3)
    xgb = XGBRegressor(max_depth=4,  # 每一棵树最大深度，默认6；
                       learning_rate=0.01587,  # 学习率，每棵树的预测结果都要乘以这个学习率，默认0.3；
                       n_estimators=2170,  # 使用多少棵树来拟合，也可以理解为多少次迭代。默认100；
                       objective='reg:linear',  # 此默认参数与 XGBClassifier 不同
                       booster='gbtree',  # 有两种模型可以选择gbtree和gblinear。gbtree使用基于树的模型进行提升计算，gblinear使用线性模型进行提升计算。默认为gbtree
                       gamma=0,  # 叶节点上进行进一步分裂所需的最小"损失减少"。默认0；
                       min_child_weight=4,  # 可以理解为叶子节点最小样本数，默认1；
                       subsample=0.4057,  # 训练集抽样比例，每次拟合一棵树之前，都会进行该抽样步骤。默认1，取值范围(0, 1]
                       colsample_bytree=0.57,  # 每次拟合一棵树之前，决定使用多少个特征，参数默认1，取值范围(0, 1]。
                       reg_alpha=0.0004,  # 默认为0，控制模型复杂程度的权重值的 L1 正则项参数，参数值越大，模型越不容易过拟合。
                       reg_lambda=0.02246)  # 默认为1，控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。

    lgbm = LGBMRegressor(objective='regression', num_leaves=5, learning_rate=0.05, n_estimators=700, max_bin=55,
                         bagging_fraction=0.8, bagging_freq=5, feature_fraction=0.25, feature_fraction_seed=9,
                         bagging_seed=9, min_data_in_leaf=6, min_sum_hessian_in_leaf=11)
    GBR = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt',
                                    min_samples_leaf=15, min_samples_split=10, loss='huber', random_state=5)
    stack_model = Model_stacking(mod=[lasso, ela, ridge, bay, svr, ker], meta_model=ker)
    y_train = y_train.values.reshape(-1, 1)
    x_train = SimpleImputer().fit_transform(x_train)
    y_train = SimpleImputer().fit_transform(y_train.reshape(-1, 1)).ravel()
    log_y_train = np.log(y_train)

    # 获取stacking第一层学习结果的堆叠特征
    x_train_stack, x_test_stack = stack_model.get_oof(x_train, log_y_train, x_test) #提取第一层模型的特征矩阵
    # 将stacking特征和数据原始的特征拼接
    x_train_add = np.hstack((x_train, x_train_stack))
    x_test_add = np.hstack((x_test, x_test_stack))
    #score = rmse_cv(stack_model,x_train_add,log_y_train)  #初步评估RMSE
    #print(score.mean())

    ## stacking最终模型预测
    # using kernelRidge as the second-layer learning model
    final_model = Model_stacking(mod=[lgbm, ela, svr, ridge, lasso, bay, xgb, GBR, ker], \
                                 meta_model=KernelRidge(alpha=0.001, kernel='polynomial', degree=1, coef0=1.8))
    final_model.fit(x_train_add, log_y_train)
    y_pred_stack = np.exp(final_model.predict(x_test_add))
    df_y_pred_stack = pd.DataFrame({'Id': x_test.index, 'SalePrice': y_pred_stack})
    score = rmse_cv(final_model, x_train_add, log_y_train).mean()
    print(score)

    '''
    # using XGBoost as the second-layer learning model
    final_model = Model_stacking(mod=[lgbm, ela, svr, ridge, lasso, bay, xgb, GBR, ker], \
                                meta_model=xgb)
    final_model.fit(x_train_add, log_y_train)
    y_pred_stack = np.exp(final_model.predict(x_test_add))
    score = rmse_cv(final_model, x_train_add, log_y_train).mean()
    print(score)  
    '''

    # save predication to submit
    df_y_pred_stack.to_csv('final_submission.csv', index=False)
    print("Your submission was successfully saved!")


if __name__ == '__main__':
    main()

