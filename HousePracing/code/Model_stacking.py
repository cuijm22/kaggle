import numpy as np
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.base import TransformerMixin

class Model_stacking(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, mod, meta_model):
        self.mod = mod  # 第一层学习器模型
        self.meta_model = meta_model  # 第二层学习器模型
        self.kf = KFold(n_splits=5, random_state=42, shuffle=True)  # 堆叠的最大特征进行K折的划分

    ## 训练函数
    def fit(self, X, y):
        self.saved_model = [list() for i in self.mod]  # 第一层所有的学习器模型
        oof_train = np.zeros((X.shape[0], len(self.mod)))  # 维度：训练样本数量*模型数量，训练集的第一层预测值

        for i, model in enumerate(self.mod):  # 索引和模型本身
            for train_index, val_index in self.kf.split(X, y):  # 数据分割成，训练集和验证集对应元素的索引
                renew_model = clone(model)
                renew_model.fit(X[train_index], y[train_index])  # 对分割出来的训练集数据进行训练
                self.saved_model[i].append(renew_model)  # 添加模型
                # oof_train[val_index,i] = renew_model.predict(X[val_index]).reshape(-1,1) #预测验证集数据

                val_prediction = renew_model.predict(X[val_index]).reshape(-1, 1)  # 验证集的预测结果

                for temp_index in range(val_prediction.shape[0]):
                    oof_train[val_index[temp_index], i] = val_prediction[temp_index]  # 预测验证集数据的目标值
        self.meta_model.fit(oof_train, y)  # 用第一层预测值作为特征，进行第二层学习器模型的训练
        return self

    ## 预测函数
    def predict(self, X):
        whole_test = np.column_stack([np.column_stack(model.predict(X) for model in single_model).mean(axis=1)
                                      for single_model in self.saved_model])  # 整个测试集第一层的预测值
        return self.meta_model.predict(whole_test)  # 第二层学习器模型对整个测试集第一层预测值特征的最终预测结果

    ## 获取第一层学习结果的堆叠特征
    def get_oof(self, X, y, test_X):
        oof = np.zeros((X.shape[0], len(self.mod)))  # 初始化
        test_single = np.zeros((test_X.shape[0], 5))  # 初始化
        # display(test_single.shape)
        test_mean = np.zeros((test_X.shape[0], len(self.mod)))
        for i, model in enumerate(self.mod):  # i是模型
            for j, (train_index, val_index) in enumerate(self.kf.split(X, y)):  # j为划分好的的数据
                clone_model = clone(model)
                clone_model.fit(X[train_index], y[train_index])
                val_prediction = clone_model.predict(X[val_index]).reshape(-1, 1)  # 验证集预测结果
                for temp_index in range(val_prediction.shape[0]):
                    oof[val_index[temp_index], i] = val_prediction[temp_index]  # 预测验证集数据

                # oof[val_index,i] = clone_model.predict(X[val_index]).reshape(-1,1)    #预测验证集

                test_prediction = clone_model.predict(test_X).reshape(-1, 1)  # 预测测试集

                test_single[:, j] = test_prediction[:, 0]
            test_mean[:, i] = test_single.mean(axis=1)  # 计算测试集均值
            return oof, test_mean
