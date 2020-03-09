## Tip1：无量纲化
无量纲化是指移除数据的单位并缩放数据到某一区间内[0-1]，以便于不同单位或者量级之间的指标可以进行比较或加权。

下面的是sklearn里的一些无量纲化的常见操作方法。

    from sklearn.datasets import load_iris  
    #导入IRIS数据集  
    iris = load_iris()
    
    #标准化，返回值为标准化后的数据  
    from sklearn.preprocessing import StandardScaler  
    StandardScaler().fit_transform(iris.data)  
    
    #区间缩放，返回值为缩放到[0, 1]区间的数据  
    from sklearn.preprocessing import MinMaxScaler  
    MinMaxScaler().fit_transform(iris.data)  
    
    #归一化，返回值为归一化后的数据
    from sklearn.preprocessing import Normalizer  
    Normalizer().fit_transform(iris.data)

## Tip2：多项式or对数的数据变换?
一个特征在当前的分布下无法有明显的区分度，但一个小小的变换则可以带来意想不到的效果。

#### 多项式变换
按照指定的degree，进行多项式操作从而衍生出新变量(当然这是针对每一列特征内的操作)。

举个栗子：

    from sklearn.datasets import load_iris  
    #导入IRIS数据集  
    iris = load_iris()
    iris.data[0]    
    # Output: array([ 5.1,  3.5,  1.4,  0.2])   
      
    tt = PolynomialFeatures().fit_transform(iris.data)  
    tt[0]    
    # Output: array([  1.  ,   5.1 ,   3.5 ,   1.4 ,   0.2 ,  26.01,  17.85,   7.14, 1.02,  12.25,   4.9 ,   0.7 ,   1.96,   0.28,   0.04])
因为PolynomialFeatures()方法默认degree是2，所以只会进行二项式的衍生。 

一般来说，多项式变换都是按照下面的方式来的：

f = kx + b 一次函数（degree为1）

f = ax^2 + b*x + w 二次函数（degree为2）

f = ax^3 + bx^2 + c*x + w 三次函数（degree为3）

这类的转换可以适当地提升模型的拟合能力，对于在线性回归模型上的应用较为广泛。

#### 对数变换
这个操作就是直接进行一个对数转换，改变原先的数据分布，而可以达到的作用主要有:

1）取完对数之后可以缩小数据的绝对数值，方便计算；

2）取完对数之后可以把乘法计算转换为加法计算；

3）还有就是分布改变带来的意想不到的效果。

numpy库里就有好几类对数转换的方法，可以通过from numpy import xxx 进行导入使用。

log：计算自然对数

log10：底为10的log

log2：底为2的log

log1p：底为e的log

#### 代码集合
    from sklearn.datasets import load_iris  
    #导入IRIS数据集  
    iris = load_iris()
    
    #多项式转换  
    #参数degree为度，默认值为2 
    from sklearn.preprocessing import PolynomialFeatures  
    PolynomialFeatures().fit_transform(iris.data)  
    
    #对数变换
    from numpy import log1p  
    from sklearn.preprocessing import FunctionTransformer  
    #自定义转换函数为对数函数的数据变换  
    #第一个参数是单变元函数  
    FunctionTransformer(log1p).fit_transform(iris.data)  

## Tip18：怎么找出数据集中有数据倾斜的特征？
今天我们用的是一个新的数据集，也是在kaggle上的一个比赛，大家可以先去下载一下：

image-20200113205105743

下载地址：https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

import pandas as pd
import numpy as np
# Plots
import seaborn as sns
import matplotlib.pyplot as plt

# 读取数据集
train = pd.read_csv('./data/house-prices-advanced-regression-techniques/train.csv')
train.head()
image-20200113210816071

我们对数据集进行分析，首先我们可以先看看特征的分布情况，看下哪些特征明显就是有数据倾斜的，然后可以找办法解决，因此，第一步就是要有办法找到这些特征。

首先可以通过可视化的方式，画箱体图，然后观察箱体情况，理论知识是：

在箱线图中，箱子的中间有一条线，代表了数据的中位数。箱子的上下底，分别是数据的上四分位数（Q3）和下四分位数（Q1），这意味着箱体包含了50%的数据。因此，箱子的高度在一定程度上反映了数据的波动程度。上下边缘则代表了该组数据的最大值和最小值。有时候箱子外部会有一些点，可以理解为数据中的“异常值”。而对于数据倾斜的，我们叫做“偏态”，与正态分布相对，指的是非对称分布的偏斜状态。在统计学上，众数和平均数之差可作为分配偏态的指标之一：如平均数大于众数，称为正偏态（或右偏态）；相反，则称为负偏态（或左偏态）。

# 丢弃y值
all_features = train.drop(['SalePrice'], axis=1)

# 找出所有的数值型变量
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric = []
for i in all_features.columns:
    if all_features[i].dtype in numeric_dtypes:
        numeric.append(i)
        
# 对所有的数值型变量绘制箱体图
sns.set_style("white")
f, ax = plt.subplots(figsize=(8, 7))
ax.set_xscale("log")
ax = sns.boxplot(data=all_features[numeric] , orient="h", palette="Set1")
ax.xaxis.grid(False)
ax.set(ylabel="Feature names")
ax.set(xlabel="Numeric values")
ax.set(title="Numeric Distribution of Features")
sns.despine(trim=True, left=True)
image-20200114215939603

可以看出有一些特征，有一些数据会偏离箱体外，因此属于数据倾斜。但是，我们从上面的可视化中虽然看出来了，但是想要选出来还是比较麻烦，所以这里引入一个偏态的概念，相对应的有一个指标skew，这个就是代表偏态的系数。

Skewness：描述数据分布形态的统计量，其描述的是某总体取值分布的对称性，简单来说就是数据的不对称程度。

偏度是三阶中心距计算出来的。

（1）Skewness = 0 ，分布形态与正态分布偏度相同。

（2）Skewness > 0 ，正偏差数值较大，为正偏或右偏。长尾巴拖在右边，数据右端有较多的极端值。

（3）Skewness < 0 ，负偏差数值较大，为负偏或左偏。长尾巴拖在左边，数据左端有较多的极端值。

（4）数值的绝对值越大，表明数据分布越不对称，偏斜程度大。

那么在Python里可以怎么实现呢？

# 找出明显偏态的数值型变量
skew_features = all_features[numeric].apply(lambda x: skew(x)).sort_values(ascending=False)

high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index

print("本数据集中有 {} 个数值型变量的 Skew > 0.5 :".format(high_skew.shape[0]))
skewness = pd.DataFrame({'Skew' :high_skew})
skew_features.head(10)
image-20200114220316028

Tip19：怎么尽可能地修正数据倾斜的特征？
上一个锦囊，分享了给大家通过skew的方法来找到数据集中有数据倾斜的特征（特征锦囊：怎么找出数据集中有数据倾斜的特征？），那么怎么去修正它呢？正是今天要分享给大家的锦囊！

还是用到房价预测的数据集：

image-20200113205105743

下载地址：https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

import pandas as pd
import numpy as np
# Plots
import seaborn as sns
import matplotlib.pyplot as plt

# 读取数据集
train = pd.read_csv('./data/house-prices-advanced-regression-techniques/train.csv')
train.head()
image-20200113210816071

我们通过上次的知识，知道了可以通过skewness来进行倾斜特征的辨别，那么对于修正它的办法，这里也先分享一个理论知识 —— box-cox转换。

线性回归模型满足线性性、独立性、方差齐性以及正态性的同时，又不丢失信息，此种变换称之为Box—Cox变换。

Box-Cox变换是Box和Cox在1964年提出的一种广义幂变换方法，是统计建模中常用的一种数据变换，用于连续的响应变量不满足正态分布的情况。Box-Cox变换之后，可以一定程度上减小不可观测的误差和预测变量的相关性。Box-Cox变换的主要特点是引入一个参数，通过数据本身估计该参数进而确定应采取的数据变换形式，Box-Cox变换可以明显地改善数据的正态性、对称性和方差相等性，对许多实际数据都是行之有效的。—— 百度百科

在使用前，我们先看看原先倾斜的特征有多少个。

# 丢弃y值
all_features = train.drop(['SalePrice'], axis=1)

# 找出所有的数值型变量
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric = []
for i in all_features.columns:
    if all_features[i].dtype in numeric_dtypes:
        numeric.append(i)
        
# 找出明显偏态的数值型变量
skew_features = all_features[numeric].apply(lambda x: skew(x)).sort_values(ascending=False)

high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index

print("本数据集中有 {} 个数值型变量的 Skew > 0.5 :".format(high_skew.shape[0]))
skewness = pd.DataFrame({'Skew' :high_skew})
skew_features
本数据集中有 24 个数值型变量的 Skew > 0.5 :

在Python中怎么使用Box-Cox 转换呢？很简单。

# 通过 Box-Cox 转换，从而把倾斜的数据进行修正
for i in skew_index:
    all_features[i] = boxcox1p(all_features[i], boxcox_normmax(all_features[i] + 1))
然后我们再看看还有多少个数据倾斜的特征吧！

# 找出明显偏态的数值型变量
skew_features = all_features[numeric].apply(lambda x: skew(x)).sort_values(ascending=False)

high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index
print("本数据集中有 {} 个数值型变量的 Skew > 0.5 :".format(high_skew.shape[0]))
skewness = pd.DataFrame({'Skew' :high_skew})
本数据集中有 15 个数值型变量的 Skew > 0.5 :

变少了很多，而且如果看他们的skew值，也会发现变小了很多。我们也可以看看转换后的箱体图情况。

# Let's make sure we handled all the skewed values
sns.set_style("white")
f, ax = plt.subplots(figsize=(8, 7))
ax.set_xscale("log")
ax = sns.boxplot(data=all_features[skew_index] , orient="h", palette="Set1")
ax.xaxis.grid(False)
ax.set(ylabel="Feature names")
ax.set(xlabel="Numeric values")
ax.set(title="Numeric Distribution of Features")
sns.despine(trim=True, left=True)
image-20200117212535708

Tip20：怎么简单使用PCA来划分数据且可视化呢？
PCA算法在数据挖掘中是很基础的降维算法，简单回顾一下定义：

PCA，全称为Principal Component Analysis，也就是主成分分析方法，是一种降维算法，其功能就是把N维的特征，通过转换映射到K维上（K<N），这些由原先N维的投射后的K个正交特征，就被称为主成分。

我们在这里使用的数据集iris，来弄一个demo：

# 导入相关库
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
%matplotlib inline

#解决中文显示问题，Mac
%matplotlib inline
from matplotlib.font_manager import FontProperties
# 设置显示的尺寸
plt.rcParams['font.family'] = ['Arial Unicode MS'] #正常显示中文

# 导入数据集
iris = load_iris()
iris_x, iris_y = iris.data, iris.target

# 实例化
pca = PCA(n_components=2)

# 训练数据
pca.fit(iris_x)
pca.transform(iris_x)[:5,]

# 自定义一个可视化的方法
label_dict = {i:k for i,k in enumerate(iris.target_names)}
def plot(x,y,title,x_label,y_label):
    ax = plt.subplot(111)
    for label,marker,color in zip(
    range(3),('^','s','o'),('blue','red','green')):
        plt.scatter(x=x[:,0].real[y == label],
                   y = x[:,1].real[y == label],
                   color = color,
                   alpha = 0.5,
                   label = label_dict[label]
                   )
        
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title(title)

# 可视化
plot(iris_x, iris_y,"原始的iris数据集","sepal length(cm)","sepal width(cm)")
plt.show()

plot(pca.transform(iris_x), iris_y,"PCA转换后的头两个正交特征","PCA1","PCA2")
image-20200129161705398

我们通过自定义的绘图函数plot，把不同类别的y值进行不同颜色的显示，从而看出在值域上分布的差异。从原始的特征来看，不同类别之间其实界限并不是十分明显，如上图所示。而进行PCA转换后，可以看出不同类别之间的界限有了比较明显的差异。

Tip21：怎么简单使用LDA来划分数据且可视化呢？
LDA算法在数据挖掘中是很基础的算法，简单回顾一下定义：

LDA的全称为Linear Discriminant Analysis, 中文为线性判别分析，LDA是一种有监督学习的算法，和PCA不同。PCA是无监督算法，。LDA是“投影后类内方差最小，类间方差最大”，也就是将数据投影到低维度上，投影后希望每一种类别数据的投影点尽可能的接近，而不同类别的数据的类别中心之间的距离尽可能的大。

我们在这里使用的数据集iris，来弄一个demo：

# 导入相关库
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
%matplotlib inline

#解决中文显示问题，Mac
%matplotlib inline
from matplotlib.font_manager import FontProperties
# 设置显示的尺寸
plt.rcParams['font.family'] = ['Arial Unicode MS'] #正常显示中文

# 导入数据集
iris = load_iris()
iris_x, iris_y = iris.data, iris.target

# 实例化
lda = LinearDiscriminantAnalysis(n_components=2)

# 训练数据
x_lda_iris = lda.fit_transform(iris_x, iris_y)


# 自定义一个可视化的方法
label_dict = {i:k for i,k in enumerate(iris.target_names)}
def plot(x,y,title,x_label,y_label):
    ax = plt.subplot(111)
    for label,marker,color in zip(
    range(3),('^','s','o'),('blue','red','green')):
        plt.scatter(x=x[:,0].real[y == label],
                   y = x[:,1].real[y == label],
                   color = color,
                   alpha = 0.5,
                   label = label_dict[label]
                   )
        
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title(title)
