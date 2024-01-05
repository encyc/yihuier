import random
import matplotlib.pyplot as plt
# plt.style.use('science')

# 变量筛选
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


class VarSelectModule:

    def __init__(self, yihui_instance):
        self.yihui_instance = yihui_instance
        self.xg_fea_imp = None
        self.rf_fea_imp = None

    # xgboost筛选变量
    def select_xgboost(self, col_list, imp_num=None):
        """
        df:数据集
        target:目标变量的字段名
        imp_num:筛选变量的个数

        return:
        xg_fea_imp:变量的特征重要性
        xg_select_col:筛选出的变量
        """
        target = self.yihui_instance.target

        x = self.yihui_instance.data[col_list].copy()
        y = self.yihui_instance.data[target]

        xgmodel = XGBClassifier(random_state=0)
        xgmodel = xgmodel.fit(x, y, eval_metric='auc')
        xg_fea_imp = pd.DataFrame({'col': list(x.columns),
                                   'imp': xgmodel.feature_importances_})
        xg_fea_imp = xg_fea_imp.sort_values('imp', ascending=False).reset_index(drop=True).iloc[:imp_num, :]
        xg_select_col = list(xg_fea_imp.col)
        return xg_fea_imp, xg_select_col

    # 随机森林筛选变量
    def select_rf(self, col_list, imp_num=None):
        """
        df:数据集
        target:目标变量的字段名
        imp_num:筛选变量的个数

        return:
        rf_fea_imp:变量的特征重要性
        rf_select_col:筛选出的变量
        """
        target = self.yihui_instance.target

        x = self.yihui_instance.data[col_list].copy()
        y = self.yihui_instance.data[target]

        rfmodel = RandomForestClassifier(random_state=0)
        rfmodel = rfmodel.fit(x, y)
        rf_fea_imp = pd.DataFrame({'col': list(x.columns),
                                   'imp': rfmodel.feature_importances_})
        rf_fea_imp = rf_fea_imp.sort_values('imp', ascending=False).reset_index(drop=True).iloc[:imp_num, :]
        rf_select_col = list(rf_fea_imp.col)
        return rf_fea_imp, rf_select_col

    # 相关性可视化
    def plot_corr(self, col_list, threshold=None, mask_direction='lt', plt_size=None, is_annot=False):
        """
        df:数据集
        col_list:变量list集合
        threshold: 相关性设定的阈值
        plt_size:图纸尺寸
        is_annot:是否显示相关系数值

        return :相关性热力图
        """
        df = self.yihui_instance.data.copy()

        corr_df = df.loc[:, col_list].corr()
        plt.figure(figsize=plt_size)
        if mask_direction == 'gt':
            sns.heatmap(corr_df, annot=is_annot, cmap='rainbow', vmax=1, vmin=-1, mask=np.abs(corr_df) >= threshold)
        if mask_direction == 'lt':
            sns.heatmap(corr_df, annot=is_annot, cmap='rainbow', vmax=1, vmin=-1, mask=np.abs(corr_df) <= threshold)
        return plt.show()

    # 相关性变量映射关系
    def corr_mapping(self, col_list, threshold=None, direction='lt'):
        """
        df:数据集
        col_list:变量list集合
        threshold: 相关性设定的阈值

        return:强相关性变量之间的映射关系表
        """
        df = self.yihui_instance.data.copy()

        corr_df = df.loc[:, col_list].corr()
        col_a = []
        col_b = []
        corr_value = []
        for col, i in zip(col_list[:-1], range(1, len(col_list), 1)):
            high_corr_col = []
            high_corr_value = []
            corr_series = corr_df[col][i:]
            for i, j in zip(corr_series.index, corr_series.values):
                if threshold == None:
                    print('Error: corr_mapping threshold is None')
                    quit()
                if direction == 'gt':
                    if abs(j) >= threshold:
                        high_corr_col.append(i)
                        high_corr_value.append(j)
                if direction == 'lt':
                    if abs(j) < threshold:
                        high_corr_col.append(i)
                        high_corr_value.append(j)
            col_a.extend([col] * len(high_corr_col))
            col_b.extend(high_corr_col)
            corr_value.extend(high_corr_value)

        corr_map_df = pd.DataFrame({'col_A': col_a,
                                    'col_B': col_b,
                                    'corr': corr_value})
        return corr_map_df

    # 相关性剔除
    def forward_delete_corr(self, col_list, threshold=None):
        """
        df:数据集
        col_list:变量list集合
        threshold: 相关性设定的阈值

        return:相关性剔除后的变量
        """
        df = self.yihui_instance.data.copy()

        list_corr = col_list[:]
        for col in list_corr:
            corr = df.loc[:, list_corr].corr()[col]
            corr_index = [x for x in corr.index if x != col]
            corr_values = [x for x in corr.values if x != 1]
            for i, j in zip(corr_index, corr_values):
                if abs(j) >= threshold:
                    list_corr.remove(i)
                    print(i,j)
        return list_corr

    # 相关性剔除（考虑IV）
    def forward_delete_corr_ivfirst(self, col_list, threshold=0.5):
        '''
        df: 数据集
        col_list: 变量list
        iv_rank: 变量的IV的list
        threshold: corr的筛选阈值

        return: 考虑了IV的大小之后，筛选出来的变量list
        '''

        df = self.yihui_instance.data.copy()

        if self.yihui_instance.binning_module.iv_df is not None:
            iv_rank = self.yihui_instance.binning_module.iv_df.sort_values(by='iv',ascending=False)
        elif self.yihui_instance.binning_module.iv_df is None:
            _, iv_rank = self.yihui_instance.binning_module.iv_num(
                col_list, max_bin=20, min_binpct=0, method='ChiMerge').sort_values(by='iv',ascending=False)


        once = self.__up_triangle(df, col_list, iv_rank=iv_rank, threshold=threshold)
        twice = self.__up_triangle(df, once, iv_rank=iv_rank, threshold=threshold)
        return twice

    def __up_triangle(self, df, col_list, iv_rank, fea_imp, threshold=0.5):
        '''
        like above
        '''
        # initial
        list_corr = col_list[:]
        # 计算变量之间的corr并存表
        corr_df = df.loc[:, col_list].corr()
        # 遍历corr_df的所有corr
        for col in list_corr:
            for row in list_corr:
                corr = corr_df.loc[row, col]
                if iv_rank is not None:
                    # 记录横纵变量的iv
                    coll = (iv_rank[iv_rank['col'] == col]['iv']).values
                    roww = (iv_rank[iv_rank['col'] == row]['iv']).values
                if fea_imp is not None:
                    # 记录横纵变量的iv
                    coll = (fea_imp[fea_imp['col'] == col]['imp']).values
                    roww = (fea_imp[fea_imp['col'] == row]['imp']).values
                # 判断
                if corr > threshold and col != row:
                    if coll > roww:
                        list_corr.remove(row)
                        # print('delete %s %s > %s' %(row,iv_col,iv_row))
                    elif coll <= roww:
                        list_corr.remove(row)
                        # print('delete %s %s > %s' %(row,iv_col,iv_row))
                        break  # 如果删除了row，则没办法继续for循环，所以要break
        return list_corr

    # 相关性剔除（考虑xgboost_imp or rf_imp）
    def forward_delete_corr_impfirst(df, col_list, fea_imp, threshold=0.5):
        '''
        df: 数据集
        col_list: 变量list
        fea_imp: 变量的imp的list,
                 可为xgboost:xg_fea_imp
                 ramdomforest:rf_fea_imp
        threshold: corr的筛选阈值

        return: 考虑了imp的大小之后，筛选出来的变量list
        '''

        def up_triangle(df, col_list, fea_imp, threshold):
            '''
            like above
            '''
            # initial
            list_corr = col_list[:]
            # 计算变量之间的corr并存表
            corr_df = df.loc[:, col_list].corr()
            # 遍历corr_df的所有corr
            for col in list_corr:
                for row in list_corr:
                    corr = corr_df.loc[row, col]
                    # 记录横纵变量的iv
                    imp_col = (fea_imp[fea_imp['col'] == col]['imp']).values
                    imp_row = (fea_imp[fea_imp['col'] == row]['imp']).values
                    # 判断
                    if corr > threshold and col != row:
                        if imp_col > imp_row:
                            list_corr.remove(row)
                            # print('delete %s %s > %s' %(row,iv_col,iv_row))
                        elif imp_col <= imp_row:
                            list_corr.remove(row)
                            # print('delete %s %s > %s' %(row,iv_col,iv_row))
                            break  # 如果删除了row，则没办法继续for循环，所以要break
            return list_corr

        once = up_triangle(df, col_list, fea_imp, threshold)
        twice = up_triangle(df, once, fea_imp, threshold)
        return twice

    # depth_first_search

    def depth_first_search(x_train, y_train, x_test, y_test, col_list, col_initial, loop_num, length):
        """
        x_train: 训练集x
        y_train: 训练集target
        x_test: 测试集x
        y_test: 测试集target
        col_list: 参与搜索的变量list
        col_initial: 参与搜索的变量list的启动list
        loop_num: 希望搜索的次数
        length: 希望搜索的变量list的长度
        """
        j = 0
        coef = []
        intercept = []
        ks_list = []
        roc_list = []
        col_func = []
        # col_pvalue_delete_list = []
        # lr_list = []
        # col_corr_delete_list = []
        # coef_col_list = []
        # lr_coe_list = []

        while j < loop_num:
            b = ''
            col = col_initial
            # model_outside = LogisticRegression()
            while len(col) < length:
                # print(len(col))
                a = random.choice(col_list)
                # print(a)
                if b == a:
                    # print('b == a')
                    a = random.choice(col_list)
                    # print(a)
                    col.append(a)
                    col = list(set(col))
                    b = a
                else:
                    # print('b <> a')
                    col.append(a)
                    col = list(set(col))
                    b = a
                # print(col)

            # 用逻辑回归训练model,默认最大迭代100次，可能会超出限制，建议多设置一点
            model = LogisticRegression(max_iter=3000)
            model.fit(x_train[col], y_train)
            # y_pred用predict_proba，因为模型整体能力较弱，基本只有0.5一下的比率
            y_pred = model.predict_proba(x_test[col])[:, 1]
            # ROC
            roc = metrics.roc_auc_score(y_test, y_pred)
            roc_list.append(roc)
            # KS
            ks_max = model_evaluation.model_ks(y_test, y_pred)
            print(col, ks_max, roc)

            '''
            #筛选p-value
            col_pvalue_delete,lr = forward_delete_pvalue(df[col],df['dlq_flag'])
            col_pvalue_delete_list.append(col_pvalue_delete)
            lr_list.append(lr)

            #筛选corr
            col_corr_delete = forward_delete_corr(df,col_list,threshold=0.6)
            col_corr_delete_list.append(col_corr_delete)

            #筛选woe系数
            coef_col,lr_coe = forward_delete_coef(df[col],df['dlq_flag'])
            coef_col_list.append(list(set(lr_coe['col'])))
            lr_coe_list.append(list(set(lr_coe['coef'])))
            '''

            # 记录coef
            coef.append(model.coef_[0])
            # 记录intercept
            intercept.append(model.intercept_[0])
            # 记录ks_max
            ks_list.append(ks_max)
            # 记录col
            col_func.append(col)

            j = j + 1
            print(j)

        ks_col_list = pd.DataFrame({'col_list': col_func,
                                    'ks_list': ks_list,
                                    'ROC': roc_list,
                                    'intercept': intercept,
                                    'coef': coef,
                                    #                            'col_pvalue_delete_list': col_pvalue_delete_list,
                                    #                            'lr_list':lr_list,
                                    #                            'col_corr_delete_list': col_corr_delete_list,
                                    #                            'coef_col_list': coef_col_list,
                                    #                            'lr_coe': lr_coe_list,
                                    })
        return ks_col_list














    # 显著性筛选,在筛选前需要做woe转换
    def forward_delete_pvalue(x_train, y_train):
        """
        x_train -- x训练集
        y_train -- y训练集

        return :显著性筛选后的变量
        """
        col_list = list(x_train.columns)
        pvalues_col = []
        for col in col_list:
            pvalues_col.append(col)
            x_train2 = sm.add_constant(x_train.loc[:, pvalues_col])
            sm_lr = sm.Logit(y_train, x_train2)
            sm_lr = sm_lr.fit()
            for i, j in zip(sm_lr.pvalues.index[1:], sm_lr.pvalues.values[1:]):
                if j >= 0.05:
                    pvalues_col.remove(i)

        x_new_train = x_train.loc[:, pvalues_col]
        x_new_train2 = sm.add_constant(x_new_train)
        lr = sm.Logit(y_train, x_new_train2)
        lr = lr.fit()
        print(lr.summary2())
        return pvalues_col, lr.summary2()

    # 逻辑回归系数符号筛选,在筛选前需要做woe转换
    def forward_delete_coef(self, x_train, y_train):
        """
        x_train -- x训练集
        y_train -- y训练集

        return :
        coef_col回归系数符号筛选后的变量
        lr_coe：每个变量的系数值
        """
        col_list = list(x_train.columns)
        coef_col = []
        for col in col_list:
            coef_col.append(col)
            x_train2 = x_train.loc[:, coef_col]
            sk_lr = LogisticRegression(random_state=0).fit(x_train2, y_train)
            coef_df = pd.DataFrame({'col': coef_col, 'coef': sk_lr.coef_[0]})
            if coef_df[coef_df.coef < 0].shape[0] > 0:
                coef_col.remove(col)

        x_new_train = x_train.loc[:, coef_col]
        lr = LogisticRegression(random_state=0).fit(x_new_train, y_train)
        lr_coe = pd.DataFrame({'col': coef_col,
                               'coef': lr.coef_[0]})
        return coef_col, lr_coe


