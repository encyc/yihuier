# Yihuier 一会儿

只需要一会儿 轻松解决逻辑回归建模

Forked https://github.com/taenggu0309/Scorecard--Function 

Encyc 修改补充重构

## 评分卡模型实现函数模块



## 函数目录：

### 1. eda.py

1. 变量的分布（可视化）

* plot_cate_var -- 类别型变量分布
* plot_num_col  -- 数值型变量分布

2. 变量的违约率分析（可视化）：

* plot_default_cate -- 类别型变量的违约率分析
* plot_default_num  -- 数值型变量的违约率分析

3. 自动EDA
* 使用ydata_profiling自动分析数据集
* 快速自动分析数据集（无图）

### 2. data_processing.py

1. 缺失值处理

* plot_bar_missing_var  -- 所有变量缺失值分布图
* plot_bar_missing_obs -- 单个样本缺失值分布图
* missing_delete_var -- 缺失值剔除（针对单个变量）
* missing_delete_obs -- 缺失值剔除（针对单个样本）
* fillna_cate_var   -- 缺失值填充（类别型变量）
* fillna_num_var    -- 缺失值填充（数值型变量）
* date_var_shift_binary -- 日期变量转换为二进制变量（日期型变量）

2. 常变量/同值化处理

* const_delete -- 常变量/同值化处理


### 3. cluster.py

* cluster_AffinityPropagation       --#亲和力传播
* cluster_AgglomerativeClustering   --#聚合
* cluster_Birch                     --#BIRCH
* cluster_DBSCAN                    --#DBSCAN
* cluster_KMeans                    --#K-Means
* cluster_MiniBatchKMeans           --#Mini-Batch K-Means
* cluster_MeanShift                 --#均值漂移
* cluster_OPTICS                    --#OPTICS
* cluster_SpectralClustering        --#光谱聚合
* cluster_GaussianMixture           --#高斯模糊



### 4.binning_funciton

* iv_count              -- 计算IV
* get_var_median        -- 关于连续变量的所有元素的中位列表
* calculate_gini        -- 计算基尼指数
* get_cart_split_point  -- 获得最优的二值划分点（即基尼指数下降最大的点）
* get_cart_bincut       --计算最优分箱切分点
* calculate_chi         --计算卡方值
* get_chimerge_bincut   --计算卡方分箱的最优分箱点
* get_maxks_split_point --计算KS值
* get_bestks_bincut     --计算最优分箱切分点
* bin_frequency         --等频分箱
* bin_distance          --等距分箱
* bin_self              --自定义分箱


### 5.binning.py

1. 分箱

* binning_cate  -- 类别型变量的分箱
* iv_cate       -- 类别型变量的IV明细表
* binning_num   -- 数值型变量的分箱（使用卡方分箱）
* iv_num        -- 数值型变量的IV明细表
* binning_self  -- 自定义分箱
* plot_woe     -- 变量woe的可视化
* woe_monoton  -- 检验变量的woe是否呈单调变化
* woe_large    -- 检验变量某个箱的woe是否过大(大于1),PS:箱体的woe在（-1,1）较合理

2. 编码

* woe_df_concat -- 变量woe结果明细表
* woe_transform -- 变量woe转换


### 6.var_select.py

* select_xgboost  -- xgboost筛选变量
* select_rf       -- 随机森林筛选变量
* plot_corr       -- 变量相关性可视化
* corr_mapping    -- 变量强相关性映射
* forward_delete_corr -- 逐个剔除相关性高的变量
* forward_delete_corr_ivfirst  -- 逐个剔除相关性高的变量（考虑IV大小）
* forward_delete_corr_impfirst  -- 逐个剔除相关性高的变量（考虑xgb或者rf）
* forward_delete_pvalue -- 显著性筛选（向前选择法）
* forward_delete_coef   -- 逻辑回归系数符号筛选（每个变量的系数符号需要一致）
* depth_first_search    -- 暴力搜索特定组合的col_list的ks，并组建叠加变量数量


### 7.model_evaluation.py

* plot_roc -- 绘制ROC曲线
* plot_model_ks -- 绘制模型的KS曲线
* plot_learning_curve -- 绘制学习曲线
* cross_verify -- 交叉验证
* plot_matrix_report -- 混淆矩阵/分类结果报告

### 8.model_implement.py

* cal_scale -- 评分卡刻度
* score_df_concat -- 变量score的明细表
* score_transform -- 变量score转换
* plot_score_ks -- 绘制评分卡的KS曲线
* plot_PR -- PR曲线
* plot_score_hist -- 好坏用户得分分布图
* score_info -- 得分明细表
* plot_lifting -- 绘制提升图和洛伦兹曲线
* rule_verify -- 设定cutoff点，计算衡量指标

### 9.model_monitor.py

* score_psi -- 计算评分的PSI
* plot_score_compare -- 评分对比图
* var_stable -- 变量稳定性分析
* plot_var_shift -- 变量偏移分析


### 10.datamerchant_tools.py

* weight_ks -- 计算加权之后的模型KS以及区间坏率