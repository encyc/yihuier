import matplotlib.pyplot as plt

# plt.style.use('science')
import numpy as np
from numpy import unique
from numpy import where
from sklearn.cluster import AffinityPropagation  # 亲和力传播
from sklearn.cluster import AgglomerativeClustering  # 聚合
from sklearn.cluster import Birch  # BIRCH
from sklearn.cluster import DBSCAN  # DBSCAN
from sklearn.cluster import KMeans  # K-Means
from sklearn.cluster import MeanShift  # 均值漂移
from sklearn.cluster import MiniBatchKMeans  # Mini-Batch K-Means
from sklearn.cluster import OPTICS  # OPTICS
from sklearn.cluster import SpectralClustering  # 光谱聚合
from sklearn.mixture import GaussianMixture  # 高斯模糊


class ClusterMuodule():
    def __init__(self, data, target):
        self.data = data
        self.target = target



    # 亲和力传播
    # cluster_AffinityPropagation(df,['v1','v2'],damping = 0.9)
    def cluster_AffinityPropagation(self, col_list, damping=0.9):
        '''
        亲和力传播包括找到一组最能概括数据的范例。
        我们设计了一种名为“亲和传播”的方法，它作为两对数据点之间相似度的输入度量。在数据点之间交换实值消息，直到一组高质量的范例和相应的群集逐渐出现。
        它是通过 AffinityPropagation 类实现的.
        要调整的主要配置是将“ 阻尼 ”设置为0.5到1，甚至可能是“首选项”。
        '''
        # 定义模型
        df = self.data
        x = np.array(df[col_list])
        model = AffinityPropagation(damping=damping)
        # 匹配模型
        print('fitting model')
        model.fit(x)
        # 为每个示例分配一个集群
        d1 = model.predict(x)
        # 检索唯一群集
        clusters = unique(d1)
        # 为每个群集的样本创建散点图
        for cluster in clusters:
            # 获取此群集的示例的行索引
            row_ix = where(d1 == cluster)
            # 创建这些样本的散布
            plt.scatter(x[row_ix, 0], x[row_ix, 1])
            # 绘制散点图
            plt.show()


    # 聚合聚类
    # cluster_AgglomerativeClustering(df,['v1','v2'],n_clusters = 2)
    def cluster_AgglomerativeClustering(df, col_list, n_clusters=2):
        '''
        聚合聚类涉及合并示例，直到达到所需的群集数量为止。
        它是层次聚类方法的更广泛类的一部分，通过AgglomerationClustering类实现的。
        主要配置是“ n _ clusters ”集，这是对数据中的群集数量的估计。
        '''
        # 定义数据集
        X = np.array(df[col_list])
        # 定义模型
        model1 = AgglomerativeClustering(n_clusters=n_clusters)
        # 模型拟合与聚类预测
        print('fitting model')
        d1 = model1.fit_predict(X)
        # 检索唯一群集
        clusters = unique(d1)
        # 为每个群集的样本创建散点图
        for cluster in clusters:
            # 获取此群集的示例的行索引
            row_ix = where(d1 == cluster)
            # 创建这些样本的散布
            plt.scatter(X[row_ix, 0], X[row_ix, 1])
            # 绘制散点图
            plt.show()


    # BIRCH聚类
    # cluster_Birch(df,['v1','v2'],threshold = 0.01, n_clusters = 2)
    def cluster_Birch(df, col_list, threshold=0.01, n_clusters=2):
        '''
        BIRCH 聚类（ BIRCH 是平衡迭代减少的缩写，聚类使用层次结构)
        包括构造一个树状结构，从中提取聚类质心。
        BIRCH 递增地和动态地群集传入的多维度量数据点，以尝试利用可用资源（即可用内存和时间约束）产生最佳质量的聚类。
        '''
        # 定义数据集
        X = np.array(df[col_list])
        model = Birch(threshold=threshold, n_clusters=n_clusters)
        # 适配模型
        print('fitting model')
        model.fit(X)
        # 为每个示例分配一个集群
        d1 = model.predict(X)
        # 检索唯一群集
        clusters = unique(d1)
        # 为每个群集的样本创建散点图
        for cluster in clusters:
            # 获取此群集的示例的行索引
            row_ix = where(d1 == cluster)
            # 创建这些样本的散布
            plt.scatter(X[row_ix, 0], X[row_ix, 1])
            # 绘制散点图
            plt.show()


    # DBSCAN聚类
    # cluster_DBSCAN(df,col_list,eps=0.30, min_samples=9)
    def cluster_DBSCAN(df, col_list, eps=0.30, min_samples=9):
        '''
        DBSCAN 聚类（其中 DBSCAN 是基于密度的空间聚类的噪声应用程序）
        涉及在域中寻找高密度区域，并将其周围的特征空间区域扩展为群集。
        我们提出了新的聚类算法 DBSCAN 依赖于基于密度的概念的集群设计，以发现任意形状的集群。
        DBSCAN 只需要一个输入参数，并支持用户为其确定适当的值
        '''
        # 定义数据集
        X = np.array(df[col_list])
        # 定义模型
        model = DBSCAN(eps=eps, min_samples=min_samples)
        # 模型拟合与聚类预测
        print('fitting model')
        d1 = model.fit_predict(X)
        # 检索唯一群集
        clusters = unique(d1)
        # 为每个群集的样本创建散点图
        for cluster in clusters:
            # 获取此群集的示例的行索引
            row_ix = where(d1 == cluster)
            # 创建这些样本的散布
            plt.scatter(X[row_ix, 0], X[row_ix, 1])
            # 绘制散点图
            plt.show()


    # K-Means聚类
    # cluster_KMeans(df,col_list,n_clusters=10)
    def cluster_KMeans(df, col_list, n_clusters=2):
        '''
        K-均值聚类可以是最常见的聚类算法，并涉及向群集分配示例，以尽量减少每个群集内的方差。
        本文的主要目的是描述一种基于样本将 N 维种群划分为 k 个集合的过程。
        这个叫做“ K-均值”的过程似乎给出了在类内方差意义上相当有效的分区。
        '''
        # 定义数据集
        X = np.array(df[col_list])
        # 定义模型
        model = KMeans(n_clusters=n_clusters)
        # 适配模型
        print('fitting model')
        model.fit(X)
        # 为每个示例分配一个集群
        d1 = model.predict(X)
        # 检索唯一群集
        clusters = unique(d1)
        # 为每个群集的样本创建散点图
        for cluster in clusters:
            # 获取此群集的示例的行索引
            row_ix = where(d1 == cluster)
            # 创建这些样本的散布
            plt.scatter(X[row_ix, 0], X[row_ix, 1])
            # 绘制散点图
            plt.show()


    # Mini-Batch K-Means聚类
    # cluster_MiniBatchKMeans(df,col_list,n_clusters = 10)
    def cluster_MiniBatchKMeans(df, col_list, n_clusters=2):
        '''
        Mini-Batch K-均值是 K-均值的修改版本。
        它使用小批量的样本而不是整个数据集对群集质心进行更新，这可以使大数据集的更新速度更快，并且可能对统计噪声更健壮。
        我们建议使用 k-均值聚类的迷你批量优化。
        与经典批处理算法相比，这降低了计算成本的数量级，同时提供了比在线随机梯度下降更好的解决方案。
        '''
        # 定义数据集
        X = np.array(df[col_list])
        # 定义模型
        model = MiniBatchKMeans(n_clusters=n_clusters)
        # 适配模型
        print('fitting model')
        model.fit(X)
        # 为每个示例分配一个集群
        d1 = model.predict(X)
        # 检索唯一群集
        clusters = unique(d1)
        # 为每个群集的样本创建散点图
        for cluster in clusters:
            # 获取此群集的示例的行索引
            row_ix = where(d1 == cluster)
            # 创建这些样本的散布
            plt.scatter(X[row_ix, 0], X[row_ix, 1])
            # 绘制散点图
            plt.show()


    # 均值飘逸聚类
    # cluster_MeanShift(df,col_list)
    def cluster_MeanShift(df, col_list):
        '''
        均值漂移聚类涉及到根据特征空间中的实例密度来寻找和调整质心。
        对离散数据证明了递推平均移位程序收敛到最接近驻点的基础密度函数，从而证明了它在检测密度模式中的应用。
        '''
        # 定义数据集
        X = np.array(df[col_list])
        # 定义模型
        model = MeanShift()
        # 适配模型
        print('fitting model')
        model.fit(X)
        # 为每个示例分配一个集群
        d1 = model.predict(X)
        # 检索唯一群集
        clusters = unique(d1)
        # 为每个群集的样本创建散点图
        for cluster in clusters:
            # 获取此群集的示例的行索引
            row_ix = where(d1 == cluster)
            # 创建这些样本的散布
            plt.scatter(X[row_ix, 0], X[row_ix, 1])
            # 绘制散点图
            plt.show()


    # OPTICS聚类
    # cluster_OPTICS(df,col_list,eps=0.8,min_samples=10)
    def cluster_OPTICS(df, col_list, eps=0.8, min_samples=10):
        '''
        OPTICS 聚类（ OPTICS 短于订购点数以标识聚类结构）是上述 DBSCAN 的修改版本。
        我们为聚类分析引入了一种新的算法，它不会显式地生成一个数据集的聚类；
        而是创建表示其基于密度的聚类结构的数据库的增强排序。
        此群集排序包含相当于密度聚类的信息，该信息对应于范围广泛的参数设置。
        '''
        # 定义数据集
        X = np.array(df[col_list])
        # 定义模型
        model = OPTICS(eps=eps, min_samples=min_samples)
        # 为每个示例分配一个集群
        print('fitting model')
        d1 = model.fit_predict(X)
        # 检索唯一群集
        clusters = unique(d1)
        # 为每个群集的样本创建散点图
        for cluster in clusters:
            # 获取此群集的示例的行索引
            row_ix = where(d1 == cluster)
            # 创建这些样本的散布
            plt.scatter(X[row_ix, 0], X[row_ix, 1])
            # 绘制散点图
            plt.show()


    # 光谱聚类
    # cluster_SpectralClustering(df,col_list,n_clusters=10)
    def cluster_SpectralClustering(df, col_list, n_clusters=2):
        '''
        光谱聚类是一类通用的聚类方法，取自线性线性代数。
        最近在许多领域出现的一个有希望的替代方案是使用聚类的光谱方法。
        这里，使用从点之间的距离导出的矩阵的顶部特征向量。
        '''
        # 定义数据集
        X = np.array(df[col_list])
        # 定义模型
        print('fitting model')
        model = SpectralClustering(n_clusters=n_clusters)
        # 为每个示例分配一个集群
        d1 = model.fit_predict(X)
        # 检索唯一群集
        clusters = unique(d1)
        # 为每个群集的样本创建散点图
        for cluster in clusters:
            # 获取此群集的示例的行索引
            row_ix = where(d1 == cluster)
            # 创建这些样本的散布
            plt.scatter(X[row_ix, 0], X[row_ix, 1])
            # 绘制散点图
            plt.show()


    # 高斯混合聚类
    # cluster_GaussianMixture(df,col_list,n_components=10)
    def cluster_GaussianMixture(df, col_list, n_components=2):
        '''
        高斯混合模型总结了一个多变量概率密度函数，顾名思义就是混合了高斯概率分布。
        '''
        # 定义数据集
        X = np.array(df[col_list])
        # 定义模型
        print('fitting model')
        model = GaussianMixture(n_components=n_components)
        # 为每个示例分配一个集群
        d1 = model.fit_predict(X)
        # 检索唯一群集
        clusters = unique(d1)
        # 为每个群集的样本创建散点图
        for cluster in clusters:
            # 获取此群集的示例的行索引
            row_ix = where(d1 == cluster)
            # 创建这些样本的散布
            plt.scatter(X[row_ix, 0], X[row_ix, 1])
            # 绘制散点图
            plt.show()



