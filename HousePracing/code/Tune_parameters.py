import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV

# 通过网格交叉验证方法，选出算法最优先验参数
class Tune_parameters:
    def __init__(self, model):
        self.model = model

    def grid_get(self, X, y, param_grid):
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring="neg_mean_squared_error")
        grid_search.fit(X, y)
        # 选出最佳参数及对应的评估指标
        print(grid_search.best_params_, np.sqrt(-grid_search.best_score_))
        grid_search.cv_results_['mean_test_score'] = np.sqrt(-grid_search.cv_results_['mean_test_score'])

        # 各参数组合对应的评估指标
        #print(pd.DataFrame(grid_search.cv_results_)[['params', 'mean_test_score', 'std_test_score']])
