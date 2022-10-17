# 机器学习

## 逻辑回归

损失函数：

$cost(h_{\theta}(x), y) = -y_{i}log(h_{\theta}(x))-(1-y_{i})log(1-h_{\theta}(x))$

其中$h_{\theta}(x) = sigmoid(W^{T}x+b)$

$sigmoid(t) = \frac{1}{(1+e^{-t})}$

上述损失函数的导数为：

$\frac{\partial L}{\partial W} = \frac{1}{m}x(\hat y-y)$

$\frac{\partial L}{\partial b} = \frac{1}{m}\sum^{m}_{i=1} (\hat y-y)$

其中$m$为batch的大小

### 手推逻辑回归的梯度

$L = $

### ⭐手撕逻辑回归

```python
import numpy as np
def sigmoid(x):
    return 1/(1+np.exp(-x))

def logistic(X, y, W, b):
    batch_size = X.shape[0]
    hidden_size = X.shape[1]

    predict = sigmoid(np.dot(X, W) + b) # y hat
    loss = -1*(y*np.log(predict) - (1-y)*np.log(1-predict))/batch_size

    dW = np.dot(X, predict-y)/batch_size
    db = np.sum(predict-y)/batch_size

    return predict, loss, dW, db

def train():    
    for i in range(epochs):
        p, loss, dW, db = logistic(X, y, W, b)
        W -= lr*dW
        b -= lr*db
```

### SVM和逻辑回归的区别和联系

SVM通过寻找最佳划分超平面来减少错误率，相应的损失函数是**hinge函数**；对数几率回归通过最大化样本输出到正确分类的概率来减少错误率，相应的损失函数是**负对数似然**。这两个损失函数的目的都是增加对分类影响较大的数据点的权重，减少与分类关系较小的数据点的权重。SVM的处理方法是只考虑 support vectors，也就是和分类最相关的少数点，去学习分类器。而逻辑回归通过非线性映射，大大减小了离分类平面较远的点的权重，相对提升了与分类最相关的数据点的权重,两者的根本目的都是一样的。

## KNN

### KNN的基本原理

决定某个点的类别时，选择离其最近的K个点进行统计，选择类别最多的作为这个点的类别。

### k的选取

选取较小的k值，意味着模型变得复杂，容易过拟合（容易学习到噪声）。

选取较大的k值，意味着模型变得简单，比如当K=样本数量时，直接预测样本数量中类别数量最多的即可。

### 特征归一化的必要性

有些特征的值会因为其自身特点比较大，在计算的时候可能就会偏向于这部分特征，所以要进行归一化。

## 正则化

### 正则化的基本概念

使用正则化项，就是给loss函数加上一个参数项。常见的是L1，L2正则化

- L0：计算非零个数，用于产生稀疏性，但是在实际研究中很少用，因为L0范数很难优化求解，是一个NP-hard问题，因此更多情况下我们是使用L1范数。
- L1：计算绝对值之和，用以产生稀疏性，因为它是L0范式的一个最优凸近似，容易优化求解。
- L2：计算平方和再开根号，L2范数更多是防止过拟合，并且让优化求解变得稳定很快速（这是因为加入了L2范式之后，满足了强凸）。

### **加入正则化项的好处**

1. 限制参数幅度，不让模型“无法无天”；
2. 限制参数搜索空间，便于收敛； 
3. 解决欠拟合与过拟合的问题。

### **什么场景用L2正则化**

只要数据线性相关，用LinearRegression拟合的不是很好，需要正则化，可以考虑使用岭回归(L2), 如果输入特征的维度很高,而且是稀疏线性关系的话， 岭回归就不太合适,考虑使用Lasso回归（L1）。

### L1 L2正则各有什么优缺点？

L2计算更方便，L1输出稀疏，会直接把不重要的特征置为0，它是一个天然的特征选择器。

### 为什么L1输出稀疏，会直接把不重要的特征置为0？

L1的梯度，在大于0时等于1，在小于0时等于-1，也就是说当L1接近零时候的梯度还是为1，所以L1会很快到0，而L2的梯度是一个二次函数，在接近0的时候梯度也会越来越小，最后不会为0。

### **weight decay**

weight decay（权值衰减）的使用其最终目的是防止过拟合。在损失函数中，weight decay是放在正则项（regularization）前面的一个系数，正则项一般指示模型的复杂度，所以weight decay的作用是调节模型复杂度对损失函数的影响，若weight decay很大，则复杂的模型损失函数的值也就大。

## 优化器

### SGD

$\theta_{t+1} = \theta_{t} - \eta*g_{t}$

$g_{t}$为$t$时刻的梯度。

更新比较频繁，会造成 cost function 有严重的震荡，最终停留在Local Minima或者Saddle Point处。

### SGD with Momentum

$\theta_{t+1} = \theta_{t} - G_{t}$

$G_{t} = \gamma*G_{t-1}+\eta * g_{t}$

$G_{t-1}$为$t-1$时刻的下降的方向

提出了动量的概念，为了跳出局部最优。使用了一阶动量， 在梯度下降的过程中加入了惯性，使得梯度方向不变的维度上速度变快，梯度方向有所改变的维度上的更新速度变慢，这样就可以加快收敛并减小震荡。一句话：SGDM不仅考虑当前梯度，也考虑到了上个时间的梯度。

### AdaGrad

  $\theta_{t+1} = \theta_{t} - G_{t}$

$G_{t} = \frac \eta {\sqrt {v_{t} + \epsilon}}*g_{t}$

$v_{t} = \sum_{i=1}^{t}g_{i}^{2}$

二阶动量(即$v_{t}$)为该维度上迄今为止所有梯度值的平方和。缺点：学习率会越来越小

### RMSProp

 $\theta_{t+1} = \theta_{t} - G_{t}$

$G_{t} = \frac \eta {\sqrt {v_{t} + \epsilon}}*g_{t}$

$v_{t} = \gamma v_{t-1} + (1-\gamma)*g_{t}^{2}$

同样是计算该维度迄今为止所有梯度值的平方和，但是时间越久影响力越低。

窗口滑动加权平均值计算二阶动量。

### Adam

$\theta_{t+1} = \theta_{t} - G_{t}$

$G_{t} = \frac \eta {\sqrt {\hat v_{t} + \epsilon}}*\hat m_{t}$

$\hat m_{t} = \frac {m_{t}} {1-\gamma_{1}}$

$\hat v_{t} = \frac {v_{t}} {1-\gamma_{2}}$

$m_{t} = \gamma_{1}*m_{t-1}+(1-\gamma_{1}) * g_{t}$

$v_{t} = \gamma_{2} v_{t-1} + (1-\gamma_{2})*g_{t}^{2}$

一阶动量+窗口滑动二阶动量

## dropout

## 激活函数

### 为什么要有激活函数？

### 激活函数有哪些，各自的优缺点是什么？

#### sigmoid函数

$sigmoid(x) = \frac 1 { 1+e^{-x}}$

![](https://pic2.zhimg.com/80/v2-83469109cd362f5fcf1decf109007fbd_1440w.png)

优点：

1. 平滑且易于求导

缺点：

1. 反向传播时导致梯度消失
2. 幂运算相对耗时
3. 输出值不以0为中心，可能会导致模型收敛速度慢

#### tanh函数

$tanh(x) = e^{x}-e^{-x}/{e^{x}+e^{-x}} $

![](https://pic2.zhimg.com/80/v2-a39596b282f6333bced6e7bfbfe04dcd_1440w.png)

特点：

解决了sigmoid函数zero-centered的问题，但是梯度消失和幂运算问题仍然存在。

#### Relu函数

$Relu(x) = max(0, x)$

![](https://pic3.zhimg.com/80/v2-5c97f377cdb5d1f0bc3faf23423c4952_1440w.png)

优点：

1. 求导快
2. 解决了梯度消失的问题
3. 收敛速度快

缺点：

1. 某些神经元可能永远不会被激活，梯度过大时会造成某些神经元死亡，当出现输出值恒小于0的时候，梯度永远为0，即参数不更新

为了解决Dead Relu Problem，学者们提出了两种激活函数：

#### **Leaky ReLU函数**

$Leaky Relu(x) = max(0.01x, x)$

![](https://pic1.zhimg.com/80/v2-8fa15614231fd01a659d4763beec9b24_1440w.png)

#### **ELU (Exponential Linear Units) 函数**

$f(x)  = x, if x>0$

$f(x) = \alpha (e^{x} - 1), otherwise$

![](https://pic2.zhimg.com/80/v2-604be114fa0478f3a1059923fd1022d1_1440w.png)

### 为什么sigmoid函数反向传播会导致梯度消失？

1. 当$x$较大时，sigmoid函数的导数会趋近于0
2. Sigmoid导数的最大值是0.25，这意味着导数在每一层至少会被压缩为原来的1/4，通过两层后被变为1/16，…，通过10层后为1/1048576。

### 为什么输出值不以0为中心时会导致模型收敛速度慢？

如果参数全为正数或负数，那么激活函数对参数的导数也全为负或正，此时模型为了收敛，不得不走“Z”字形逼近最优解。模型存在多个参数，如果导数全为负或全为正，这些参数都只能往一个方向走，就是“Z”字形。

## 决策树

### 决策树如何划分

#### 信息增益

“信息熵”是度量样本集合纯度的最常用的指标。假定当前样本集合$D$中第$k$类样本所占的比例为$p_k$，则$D$的信息熵定义为

$Ent(D) = -\sum_{k=1}^{n}p_klog_{2}p_k$

其中n为标签的种类。

当我们进行一次划分时，可以计算该划分所获得的“信息增益”：

$Gain(D, a) = Ent(d)-\sum_{v=1}^{V}frac*Ent(D^{v})$

这条公式的意思是，如果我们选择$a$属性进行划分所能带来的信息增益是多少，其中$Ent(D^{v})$代表某个子结点内的信息熵。

一般而言，信息增益越大，则意味着用属性$a$来进行划分所获得的“纯度提升”越大。著名的ID3决策树就是以信息增益为准则来划分属性。

#### 增益率

信息增益准则对可取值数目较多的属性有所偏好（因为属性种类越多的话，信息增益大的可能性会大，有点像过拟合），为减少这种偏好可能带来的不利影响，C4.5决策树采用了增益率，而不是信息增益。

$Gain\_ratio(D, a) = \frac{Gain(D, a)}{IV(a)}$

其中$IV(a)$为该属性各个种类概率值的信息熵，也就是种类越多，增益率会越小，有点像正则项。

#### 基尼指数

CART决策树使用“基尼指数”来选择划分属性：

$Gini(D) = 1 - \sum_{k=1}^{n}p_k^2$

直观上来说，基尼指数反映了从数据集中随机抽取两个样本，其类别标记不一致的概率，因此，基尼指数越小，则数据集D的纯度越高。

### 决策树如何剪枝

#### 预剪枝

预剪枝是指在决策树生成过程中，对每个节点在划分前先进行估计，若当前结点的划分不能带来决策树泛化性能的提升（在验证集上的效果），则停止划分并将当前结点标记为叶节点。

预剪枝能降低过拟合的风险，还显著减少了决策树的训练时间开销和测试时间开销。

#### 后剪枝

后剪枝是先从训练集生成一棵完整的决策树，然后自底向上地对非叶节点进行考察，若将该节点对应的子树替换为叶节点能带来泛化性能的提升，则将该子树替换为叶节点。

后剪枝通常比预剪枝保留了更多的分支，一般情形下，后剪枝的欠拟合风险很小，泛化性能往往优于预剪枝，但其训练时间开销比不剪枝和预剪枝都要大得多。

### 决策树如何处理连续值和缺失值

#### 连续值

最简单的策略是采用二分法，即将连续属性划分成0和1，可以基于信息增益等准则像决策树划分一样去寻找一个最好的划分点。

#### 缺失值

我们需要解决两个问题：

1. 如何在属性值缺失的情况下进行划分属性选择？（也就是如何根据训练集构建决策树）
2. 若样本在该属性上的值缺失，如何对样本进行划分？

对于问题1， 

对于问题2，可以让缺失值样本以不同的概率划分到不同的子结点中去。

### CART

https://www.cnblogs.com/pinard/p/6053344.html

# 集成学习

集成学习通过构建并结合多个学习器来完成学习任务，要想获得好的集成，个体学习器应“好而不同”，即个体学习器要有一定的“准确性”，即学习器不能太坏，并且要有“多样性”，即学习器间具有差异。

## Boosting

### boosting基本概念

boosting是一族可以将弱学习器提升为强学习器的算法，先从初始训练集中训练出一个基学习器，再根据基学习器的表现对训练样本分布进行调整，使得先前基学习器做错的训练样本在后续得到更多关注，然后基于调整后的样本分布来训练下一个基分类器，重复进行，直到基学习器数目达到实现指定的值T。最后加权求和。boosting算法主要关注降低偏差。

### AdaBoost

参考：https://zhuanlan.zhihu.com/p/42740654

https://www.cnblogs.com/pinard/p/6133937.html

AdaBoost算法有多种推导方式，比较容易理解的是“加法模型”，即基学习器的线性组合：

$H(x) = \sum_{t=1}^{T}\alpha_{t}h_{t}(x)$

通过改变训练集样本的权重（使上一次分错的样本在下一次训练中权重更高）得到新的学习器。

AdaBoost的流程如下：

![](https://pic2.zhimg.com/80/v2-bf2a3f8490d6f2949b112d655babc749_720w.jpg)

在每一次迭代时，需要先计算误差率（第4行），再用误差率计算$\alpha_t$（第5行），$\alpha_t$既被用来当作第t个学习器的权重，也被用来更新训练集样本的权重（第8行）。

$\alpha_t$是通过对损失函数求导并且使其等于0得到的，AdaBoost的损失函数为指数函数：

![](https://pic3.zhimg.com/80/v2-e90418bb36190d457d081d341af4c21e_720w.jpg)

Adaboost的主要优点有：

1. Adaboost作为分类器时，分类精度很高

　2. 在Adaboost的框架下，可以使用各种回归分类模型来构建弱学习器，非常灵活。

　3. 作为简单的二元分类器时，构造简单，结果可理解。

　4. 不容易发生过拟合

Adaboost的主要缺点有：

　1. 对异常样本敏感，异常样本在迭代中可能会获得较高的权重，影响最终的强学习器的预测准确性。

### GBDT

https://www.cnblogs.com/pinard/p/6140514.html

### XGBOOST

https://www.cnblogs.com/pinard/p/10979808.html



## xgboost面试准备

### **1. 简单介绍一下XGBoost** 

首先需要说一说GBDT，它是一种基于boosting增强策略的加法模型，训练的时候采用前向分布算法进行贪婪的学习，每次迭代都学习一棵CART树来拟合之前 t-1 棵树的预测结果与训练样本真实值的残差。

XGBoost对GBDT进行了一系列优化，比如损失函数进行了二阶泰勒展开、目标函数加入正则项、支持并行和默认缺失值处理等，在可扩展性和训练速度上有了巨大的提升，但其核心思想没有大的变化。

### **2. XGBoost与GBDT有什么不同**

- **基分类器**：XGBoost的基分类器不仅支持CART决策树，还支持线性分类器，此时XGBoost相当于带L1和L2正则化项的Logistic回归（分类问题）或者线性回归（回归问题）。
- **导数信息**：XGBoost对损失函数做了二阶泰勒展开，GBDT只用了一阶导数信息，并且XGBoost还支持自定义损失函数，只要损失函数一阶、二阶可导。
- **正则项**：XGBoost的目标函数加了正则项， 相当于预剪枝，使得学习出来的模型更加不容易过拟合。
- **列抽样**：XGBoost支持列采样，与随机森林类似，用于防止过拟合。
- **缺失值处理**：对树中的每个非叶子结点，XGBoost可以自动学习出它的默认分裂方向。如果某个样本该特征值缺失，会将其划入默认分支。
- **并行化**：注意不是tree维度的并行，而是特征维度的并行。XGBoost预先将每个特征按特征值排好序，存储为块结构，分裂结点时可以采用多线程并行查找每个特征的最佳分割点，极大提升训练速度。

### **3. XGBoost为什么使用泰勒二阶展开**

xgboost是用二阶泰勒展开的优势在哪？ - 知乎 https://www.zhihu.com/question/61374305

- **精准性**：相对于GBDT的一阶泰勒展开，XGBoost采用二阶泰勒展开，可以更为精准的逼近真实的损失函数
- **可扩展性**：损失函数支持自定义，只需要新的损失函数二阶可导。

### **4. XGBoost为什么可以并行训练**

- XGBoost的并行，并不是说每棵树可以并行训练，XGB本质上仍然采用boosting思想，每棵树训练前需要等前面的树训练完成才能开始训练。
- XGBoost的并行，指的是特征维度的并行：在训练之前，每个特征按特征值对样本进行预排序，并存储为Block结构，在后面查找特征分割点时可以重复使用，而且特征已经被存储为一个个block结构，那么在寻找每个特征的最佳分割点时，可以利用多线程对每个block[并行计算](https://cloud.tencent.com/product/gpu?from=10680)。

### **5. XGBoost为什么快**

- **分块并行**：训练前每个特征按特征值进行排序并存储为Block结构，后面查找特征分割点时重复使用，并且支持并行查找每个特征的分割点
- **候选分位点**：每个特征采用常数个分位点作为候选分割点
- **CPU cache 命中优化**： 使用缓存预取的方法，对每个线程分配一个连续的buffer，读取每个block中样本的梯度信息并存入连续的Buffer中。
- **Block 处理优化**：Block预先放入内存；Block按列进行解压缩；将Block划分到不同硬盘来提高吞吐

### **6. XGBoost防止过拟合的方法**

XGBoost在设计时，为了防止过拟合做了很多优化，具体如下：

- **目标函数添加正则项**：叶子节点个数+叶子节点权重的L2正则化
- **列抽样**：训练的时候只用一部分特征（不考虑剩余的block块即可）
- **子采样**：每轮计算可以不使用全部样本，使算法更加保守
- **shrinkage**: 可以叫学习率或步长，为了给后面的训练留出更多的学习空间

### **7. XGBoost如何处理缺失值**

XGBoost模型的一个优点就是允许特征存在缺失值。对缺失值的处理方式如下：

- 在特征k上寻找最佳 split point 时，不会对该列特征 missing 的样本进行遍历，而只对该列特征值为 non-missing 的样本上对应的特征值进行遍历，通过这个技巧来减少了为稀疏离散特征寻找 split point 的时间开销。
- 在逻辑实现上，为了保证完备性，会将该特征值missing的样本分别分配到左叶子结点和右叶子结点，两种情形都计算一遍后，选择分裂后增益最大的那个方向（左分支或是右分支），作为预测时特征值缺失样本的默认分支方向。
- 如果在训练中没有缺失值而在预测中出现缺失，那么会自动将缺失值的划分方向放到右子结点。

![img](https://ask.qcloudimg.com/http-save/yehe-1622140/x5jfcrynnp.jpeg?imageView2/2/w/1620)

find_split时，缺失值处理的伪代码

### **8. XGBoost中叶子结点的权重如何计算出来**

XGBoost目标函数最终推导形式如下：

![img](https://ask.qcloudimg.com/http-save/yehe-1622140/19t9mjhjek.jpeg?imageView2/2/w/1620)

利用一元二次函数求最值的知识，当目标函数达到最小值Obj*时，每个叶子结点的权重为wj*。

具体公式如下：

![img](https://ask.qcloudimg.com/http-save/yehe-1622140/6wmb1abb3v.jpeg?imageView2/2/w/1620)

### **9. XGBoost中的一棵树的停止生长条件**

- 当新引入的一次分裂所带来的增益Gain<0时，放弃当前的分裂。这是训练损失和模型结构复杂度的博弈过程。
- 当树达到最大深度时，停止建树，因为树的深度太深容易出现过拟合，这里需要设置一个超参数max_depth。
- 当引入一次分裂后，重新计算新生成的左、右两个叶子结点的样本权重和。如果任一个叶子结点的样本权重低于某一个阈值，也会放弃此次分裂。这涉及到一个超参数:最小样本权重和，是指如果一个叶子节点包含的样本数量太少也会放弃分裂，防止树分的太细。

### **10. RF和GBDT的区别**

**相同点：**

- 都是由多棵树组成，最终的结果都是由多棵树一起决定。

**不同点：**

- **集成学习**：RF属于bagging思想，而GBDT是boosting思想
- **偏差-方差权衡**：RF不断的降低模型的方差，而GBDT不断的降低模型的偏差
- **训练样本**：RF每次迭代的样本是从全部训练集中有放回抽样形成的，而GBDT每次使用全部样本
- **并行性**：RF的树可以并行生成，而GBDT只能顺序生成(需要等上一棵树完全生成)
- **最终结果**：RF最终是多棵树进行多数表决（回归问题是取平均），而GBDT是加权融合
- **数据敏感性**：RF对异常值不敏感，而GBDT对异常值比较敏感
- **泛化能力**：RF不易过拟合，而GBDT容易过拟合

### **11. XGBoost如何处理不平衡数据**

对于不平衡的数据集，例如用户的购买行为，肯定是极其不平衡的，这对XGBoost的训练有很大的影响，XGBoost有两种自带的方法来解决：

第一种，如果你在意AUC，采用AUC来评估模型的性能，那你可以通过设置scale_pos_weight来平衡正样本和负样本的权重。例如，当正负样本比例为1:10时，scale_pos_weight可以取10；

第二种，如果你在意概率(预测得分的合理性)，你不能重新平衡数据集(会破坏数据的真实分布)，应该设置max_delta_step为一个有限数字来帮助收敛（基模型为LR时有效）。

原话是这么说的：

```javascript
For common cases such as ads clickthrough log, the dataset is extremely imbalanced. This can affect the training of xgboost model, 
and there are two ways to improve it.
  If you care only about the ranking order (AUC) of your prediction
      Balance the positive and negative weights, via scale_pos_weight
      Use AUC for evaluation
  If you care about predicting the right probability
      In such a case, you cannot re-balance the dataset
      In such a case, set parameter max_delta_step to a finite number (say 1) will help convergence
```

复制

那么，源码到底是怎么利用**scale_pos_weight**来平衡样本的呢，是调节权重还是过采样呢？请看源码：

```javascript
if (info.labels[i] == 1.0f)  w *= param_.scale_pos_weight
```

复制

可以看出，应该是增大了少数样本的权重。

除此之外，还可以通过上采样、下采样、SMOTE算法或者自定义代价函数的方式解决正负样本不平衡的问题。

### **12. 比较LR和GBDT，说说什么情景下GBDT不如LR**

先说说LR和GBDT的区别：

- LR是线性模型，可解释性强，很容易并行化，但学习能力有限，需要大量的人工特征工程
- GBDT是非线性模型，具有天然的特征组合优势，特征表达能力强，但是树与树之间无法并行训练，而且树模型很容易过拟合；

当在高维稀疏特征的场景下，LR的效果一般会比GBDT好。原因如下：

先看一个例子：

> 假设一个二分类问题，label为0和1，特征有100维，如果有1w个样本，但其中只要10个正样本1，而这些样本的特征 f1的值为全为1，而其余9990条样本的f1特征都为0(在高维稀疏的情况下这种情况很常见)。 我们都知道在这种情况下，树模型很容易优化出一个使用f1特征作为重要分裂节点的树，因为这个结点直接能够将训练数据划分的很好，但是当测试的时候，却会发现效果很差，因为这个特征f1只是刚好偶然间跟y拟合到了这个规律，这也是我们常说的过拟合。

那么这种情况下，如果采用LR的话，应该也会出现类似过拟合的情况呀：y = W1*f1 + Wi*fi+….，其中 W1特别大以拟合这10个样本。为什么此时树模型就过拟合的更严重呢？

仔细想想发现，因为现在的模型普遍都会带着正则项，而 LR 等线性模型的正则项是对权重的惩罚，也就是 W1一旦过大，惩罚就会很大，进一步压缩 W1的值，使他不至于过大。但是，树模型则不一样，树模型的惩罚项通常为叶子节点数和深度等，而我们都知道，对于上面这种 case，树只需要一个节点就可以完美分割9990和10个样本，一个结点，最终产生的惩罚项极其之小。

这也就是为什么在高维稀疏特征的时候，线性模型会比非线性模型好的原因了：**带正则化的线性模型比较不容易对稀疏特征过拟合。**

### **13. XGBoost中如何对树进行剪枝**

- 在目标函数中增加了正则项：使用叶子结点的数目和叶子结点权重的L2模的平方，控制树的复杂度。
- 在结点分裂时，定义了一个阈值，如果分裂后目标函数的增益小于该阈值，则不分裂。
- 当引入一次分裂后，重新计算新生成的左、右两个叶子结点的样本权重和。如果任一个叶子结点的样本权重低于某一个阈值（最小样本权重和），也会放弃此次分裂。
- XGBoost 先从顶到底建立树直到最大深度，再从底到顶反向检查是否有不满足分裂条件的结点，进行剪枝。

### **14. XGBoost如何选择最佳分裂点？** 

XGBoost在训练前预先将特征按照特征值进行了排序，并存储为block结构，以后在结点分裂时可以重复使用该结构。

因此，可以采用特征并行的方法利用多个线程分别计算每个特征的最佳分割点，根据每次分裂后产生的增益，最终选择增益最大的那个特征的特征值作为最佳分裂点。

如果在计算每个特征的最佳分割点时，对每个样本都进行遍历，计算复杂度会很大，这种全局扫描的方法并不适用[大数据](https://cloud.tencent.com/solution/bigdata?from=10680)的场景。XGBoost还提供了一种直方图近似算法，对特征排序后仅选择常数个候选分裂位置作为候选分裂点，极大提升了结点分裂时的计算效率。

### **15. XGBoost的Scalable性如何体现**

- **基分类器的scalability**：弱分类器可以支持CART决策树，也可以支持LR和Linear。
- **目标函数的scalability**：支持自定义loss function，只需要其一阶、二阶可导。有这个特性是因为泰勒二阶展开，得到通用的目标函数形式。
- **学习方法的scalability**：Block结构支持并行化，支持 Out-of-core计算。

### **16. XGBoost如何评价特征的重要性**

我们采用三种方法来评判XGBoost模型中特征的重要程度：

- **weight** ：该特征在所有树中被用作分割样本的特征的总次数。
- **gain** ：该特征在其出现过的所有树中产生的平均增益。
- **cover** ：该特征在其出现过的所有树中的平均覆盖范围。

> 注意：覆盖范围这里指的是一个特征用作分割点后，其影响的样本数量，即有多少样本经过该特征分割到两个子节点。

### **17. XGBooost参数调优的一般步骤**

首先需要初始化一些基本变量，例如：

- max_depth = 5
- min_child_weight = 1
- gamma = 0
- subsample, colsample_bytree = 0.8
- scale_pos_weight = 1

**(1) 确定learning rate和estimator的数量**

learning rate可以先用0.1，用cv来寻找最优的estimators

**(2) max_depth和 min_child_weight**

我们调整这两个参数是因为，这两个参数对输出结果的影响很大。我们首先将这两个参数设置为较大的数，然后通过迭代的方式不断修正，缩小范围。

max_depth，每棵子树的最大深度，check from range(3,10,2)。

min_child_weight，子节点的权重阈值，check from range(1,6,2)。

如果一个结点分裂后，它的所有子节点的权重之和都大于该阈值，该叶子节点才可以划分。

**(3) gamma**

也称作最小划分损失`min_split_loss`，check from 0.1 to 0.5，指的是，对于一个叶子节点，当对它采取划分之后，损失函数的降低值的阈值。

- 如果大于该阈值，则该叶子节点值得继续划分
- 如果小于该阈值，则该叶子节点不值得继续划分

**(4) subsample, colsample_bytree**

subsample是对训练的采样比例

colsample_bytree是对特征的采样比例

both check from 0.6 to 0.9

**(5) 正则化参数**

alpha 是L1正则化系数，try 1e-5, 1e-2, 0.1, 1, 100

lambda 是L2正则化系数

**(6) 降低学习率**

降低学习率的同时增加树的数量，通常最后设置学习率为0.01~0.1

### **18. XGBoost模型如果过拟合了怎么解决**

当出现过拟合时，有两类参数可以缓解：

第一类参数：用于直接控制模型的复杂度。包括`max_depth,min_child_weight,gamma` 等参数

第二类参数：用于增加随机性，从而使得模型在训练时对于噪音不敏感。包括`subsample,colsample_bytree`

还有就是直接减小`learning rate`，但需要同时增加`estimator` 参数。

### **19.为什么XGBoost相比某些模型对缺失值不敏感**

对存在缺失值的特征，一般的解决方法是：

- 离散型变量：用出现次数最多的特征值填充；
- 连续型变量：用中位数或均值填充；

一些模型如SVM和KNN，其模型原理中涉及到了对样本距离的度量，如果缺失值处理不当，最终会导致模型预测效果很差。

而树模型对缺失值的敏感度低，大部分时候可以在数据缺失时时使用。原因就是，一棵树中每个结点在分裂时，寻找的是某个特征的最佳分裂点（特征值），完全可以不考虑存在特征值缺失的样本，也就是说，如果某些样本缺失的特征值缺失，对寻找最佳分割点的影响不是很大。

XGBoost对缺失数据有特定的处理方法，[详情参考上篇文章第7题](http://mp.weixin.qq.com/s?__biz=Mzg2MjI5Mzk0MA==&mid=2247484181&idx=1&sn=8d0e51fb0cb974f042e66659e1daf447&chksm=ce0b59cef97cd0d8cf7f9ae1e91e41017ff6d4c4b43a4c19b476c0b6d37f15769f954c2965ef&scene=21#wechat_redirect)。

因此，对于有缺失值的数据在经过缺失处理后：

- 当数据量很小时，优先用朴素贝叶斯
- 数据量适中或者较大，用树模型，优先XGBoost
- 数据量较大，也可以用神经网络
- 避免使用距离度量相关的模型，如KNN和SVM

### **20. XGBoost和LightGBM的区别**

![img](https://ask.qcloudimg.com/http-save/yehe-1622140/btc3oj2txs.jpeg?imageView2/2/w/1620)

（1）树生长策略：XGB采用`level-wise`的分裂策略，LGB采用`leaf-wise`的分裂策略。XGB对每一层所有节点做无差别分裂，但是可能有些节点增益非常小，对结果影响不大，带来不必要的开销。Leaf-wise是在所有叶子节点中选取分裂收益最大的节点进行的，但是很容易出现过拟合问题，所以需要对最大深度做限制 。

（2）分割点查找算法：XGB使用特征预排序算法，LGB使用基于直方图的切分点算法，其优势如下：

- 减少内存占用，比如离散为256个bin时，只需要用8位整形就可以保存一个样本被映射为哪个bin(这个bin可以说就是转换后的特征)，对比预排序的exact greedy算法来说（用int_32来存储索引+ 用float_32保存特征值），可以节省7/8的空间。
- 计算效率提高，预排序的Exact greedy对每个特征都需要遍历一遍数据，并计算增益，复杂度为?(#???????×#????)。而直方图算法在建立完直方图后，只需要对每个特征遍历直方图即可，复杂度为?(#???????×#????)。
- LGB还可以使用直方图做差加速，一个节点的直方图可以通过父节点的直方图减去兄弟节点的直方图得到，从而加速计算

> 但实际上xgboost的近似直方图算法也类似于lightgbm这里的直方图算法，为什么xgboost的近似算法比lightgbm还是慢很多呢？ xgboost在每一层都动态构建直方图， 因为xgboost的直方图算法不是针对某个特定的feature，而是所有feature共享一个直方图(每个样本的权重是二阶导)，所以每一层都要重新构建直方图，而lightgbm中对每个特征都有一个直方图，所以构建一次直方图就够了。

（3）支持离散变量：无法直接输入类别型变量，因此需要事先对类别型变量进行编码（例如独热编码），而LightGBM可以直接处理类别型变量。

（4）缓存命中率：XGB使用Block结构的一个缺点是取梯度的时候，是通过索引来获取的，而这些梯度的获取顺序是按照特征的大小顺序的，这将导致非连续的内存访问，可能使得CPU cache缓存命中率低，从而影响算法效率。而LGB是基于直方图分裂特征的，梯度信息都存储在一个个bin中，所以访问梯度是连续的，缓存命中率高。

（5）LightGBM 与 XGboost 的并行策略不同：

- **特征并行** ：LGB特征并行的前提是每个worker留有一份完整的数据集，但是每个worker仅在特征子集上进行最佳切分点的寻找；worker之间需要相互通信，通过比对损失来确定最佳切分点；然后将这个最佳切分点的位置进行全局广播，每个worker进行切分即可。XGB的特征并行与LGB的最大不同在于XGB每个worker节点中仅有部分的列数据，也就是垂直切分，每个worker寻找局部最佳切分点，worker之间相互通信，然后在具有最佳切分点的worker上进行节点分裂，再由这个节点广播一下被切分到左右节点的样本索引号，其他worker才能开始分裂。二者的区别就导致了LGB中worker间通信成本明显降低，只需通信一个特征分裂点即可，而XGB中要广播样本索引。
- **数据并行** ：当数据量很大，特征相对较少时，可采用数据并行策略。LGB中先对数据水平切分，每个worker上的数据先建立起局部的直方图，然后合并成全局的直方图，采用直方图相减的方式，先计算样本量少的节点的样本索引，然后直接相减得到另一子节点的样本索引，这个直方图算法使得worker间的通信成本降低一倍，因为只用通信以此样本量少的节点。XGB中的数据并行也是水平切分，然后单个worker建立局部直方图，再合并为全局，不同在于根据全局直方图进行各个worker上的节点分裂时会单独计算子节点的样本索引，因此效率贼慢，每个worker间的通信量也就变得很大。
- **投票并行（LGB）**：当数据量和维度都很大时，选用投票并行，该方法是数据并行的一个改进。数据并行中的合并直方图的代价相对较大，尤其是当特征维度很大时。大致思想是：每个worker首先会找到本地的一些优秀的特征，然后进行全局投票，根据投票结果，选择top的特征进行直方图的合并，再寻求全局的最优分割点。

## Bagging

### bagging基本概念

给定包含m个样本的数据集，放回的进行m次随机采样，会得到一个含m个样本的采样集。按照这样的采样方式，采样T次，得到T个不同的采样集，基于每个采样集训练出一个基学习器，对分类任务采用简单投票法，对回归任务使用简单平均法。bagging算法主要关注降低方差。

### 随机森林

随机森林是Bagging算法的一个变种，它在以决策树为基学习器构建Bagging集成的基础上，进一步在决策树的训练过程引入了随机属性选择。跟dropout有着异曲同工之妙。





- 

# 词向量

## word2vec

https://www.cnblogs.com/pinard/p/7160330.html

### CBOW与Skip-Gram

用词袋模型训练

CBOW：用一个词的上下文去预测该词

Skip-Gram：用一个词去预测它的上下文（取Softmax前n个词）

### 分层Softmax

https://www.cnblogs.com/pinard/p/7243513.html

用哈夫曼编码树代替原来的softmax，每层做一次二分类。

优点：提高模型训练的效率

缺点：如果预测的词是生僻词，还是会很“辛苦”

### 负采样

https://www.cnblogs.com/pinard/p/7249903.html

用负采样采样出Negative Samples,在正例和负例之间做二元逻辑回归。

# transformer

## transformer的结构

<img src="https://pic1.zhimg.com/80/v2-4b53b731a961ee467928619d14a5fd44_1440w.jpg" style="zoom: 50%;" />

简单可分成：

- Position embedding
- Encoder Multi-head Attention
- Decoder Multi-head Attention
- Encoder-decoder multi-head attention
- Add&Norm
- FeedForward Network

## Multi-head Attention

掌握程度：手撕

```python
# 官方实现版，用于理解
class MultiHeadedAttention(nn.Module):
	def __init__(self, h, d_model, dropout=0.1):
		super(MultiHeadedAttention, self).__init__()
		assert d_model % h == 0
		# We assume d_v always equals d_k
		self.d_k = d_model // h
		self.h = h
		self.linears = clones(nn.Linear(d_model, d_model), 4)
		self.attn = None
		self.dropout = nn.Dropout(p=dropout)
	
	def forward(self, query, key, value, mask=None): 
		if mask is not None:
			# 所有h个head的mask都是相同的 
			mask = mask.unsqueeze(1)
		nbatches = query.size(0)
		
		# 1) 首先使用线性变换，然后把d_model分配给h个Head，每个head为d_k=d_model/h 
		query, key, value = \
			[l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)	
				for l, x in zip(self.linears, (query, key, value))]
		
		# 2) 使用attention函数计算
		x, self.attn = attention(query, key, value, mask=mask, 
			dropout=self.dropout)
		
		# 3) 把8个head的64维向量拼接成一个512的向量。然后再使用一个线性变换(512,521)，shape不变。 
		x = x.transpose(1, 2).contiguous() \
			.view(nbatches, -1, self.h * self.d_k)
		return self.linears[-1](x)
```

```python
# MHA手撕版本
def attention(q, k, v):
    head = 8
    dim_per_head = hidden_size // head
 
    def head_shape(x):
        return x.reshape(batch_size, -1, head, dim_per_head).transpose(1, 2)

    def head_unshape(x):
        return x.transpose(1, 2).contiguous().reshape(batch_size, -1, head * dim_per_head)

    q = head_shape(q)
    k = head_shape(k)
    v = head_shape(v)

    query = linear_querys(q)
    key = linear_keys(k)

    query /= math.sqrt(dim_per_head)
    attn = torch.matual(q, k.transpose(2, 3))
    attn = softmax(attn)

    res = torch.matual(attn, v)
    res = head_unshape(res)

    return res
```

### 为什么transformer的decoder有masked multi-head attention，有什么区别？意义？

是因为在decode的时候预测第i个词的时候遮住i后面的词。

### transformer除了decoder有mask，还有哪里做了mask操作？为什么？

整个Transformer中包含三种类型的attention,且目的并不相同。

- Encoder的self-attention，考虑到batch的并行化，通常会进行padding，因此会对序列中mask=0的token进行mask后在进行attention score的softmax归一化。
- Decoder中的self-attention，为了避免预测时后续tokens的影所以必须令后续tokens的mask=0，其具体做法为构造一个三角矩阵。
- Decoder中的encode-decoder attention，涉及到decoder中当前token与整个encoder的sequence的计算，所以encoder仍然需要考虑mask。

综上，无论对于哪个类型的attention，在进行sotmax归一化前，都需要考虑mask操作。

### 为什么要multi-head?

多头可以使参数矩阵形成多个子空间，矩阵整体的size不变，只是改变了每个head对应的维度大小，这样做使矩阵对多方面信息进行学习，但是计算量和单个head差不多。

### Decoder阶段的mha和encoder的mha有什么区别？

Decoder有两层mha，encoder有一层mha，Decoder的第二层mha是为了转化输入与输出句长，Decoder的请求q与键k和数值v的倒数第二个维度可以不一样，但是encoder的qkv维度一样。

> 两者的第一层` MHA 都是 self-attention。Decoder 的第二层是 cross-attention，目的是让 target 去 attend source。
>
> 至于维度，只是因为 tgt 和 src 长度不同而已。（假设 `batch_first=True`）Q/K/V 维度都是 (N, L, E)，其中 L 是 序列长度。 self-attention 是自己 attend 自己，所以 Q/K/V 对应的 L 都是 S=len(src) 或者 T=len(tgt)。cross-attention 是自己 attend 别人，所以 Q 的 L 是自己的 L，K/V 的 L 是别人的 L，即 cross-attention 的时候 Q.size()=(N, T, E), K.size()=(N, S, E), V.size()=(N, S, E)。
>
> （LLJ 补充于 2022-04-21）

### bert的mask为何不学习transformer在attention处进行屏蔽score的技巧？

BERT和transformer的目标不一致，bert是语言的预训练模型，需要充分考虑上下文的关系，而transformer主要考虑句子中第i个元素与前i-1个元素的关系。

### 为什么使用不同的Q和K？

是为了打破对称性，参考其中“如果令Q=K，那么得到的模型大概率会得到一个类似单位矩阵的attention矩阵，**这样self-attention就退化成一个point-wise线性映射**。这样至少是违反了设计的初衷。”

知乎 https://www.zhihu.com/question/319339652https://www.bilibili.com/read/cv4902832

补充得到单位矩阵的原因：attention矩阵中元素 (i, j) 是向量 i 与向量 j 点乘后、softmax 后的结果。显然，加入所有向量长度相同，那么“自己点乘自己”肯定是比“自己点乘别人”要大的。

### 为什么要除以delta dk，dk的含义是？

向量的点积结果会很大，将softmax函数的梯度就会很小，scaled会缓解这种现象。在输入的数量级很大时，softmax的梯度会消失为0， 造成参数更新困难。dk是Q\*K的方差，方差越大说明点积的数量级越大，在除以根号dk后方差变为1，也就有效地控制了前面提到的梯度消失的问题。

### 怎么估计self-attention的时间复杂度？

假设$Q$和$K$的维度为$n*h$

$Q\cdot K$的复杂度为$O(n^{2}d)$

对$n*n$做softmax，复杂度为$O(n^2)$

$(n*n) \cdot (n*d)$的复杂度为$O(n^2d)$

总体复杂度是 $O(n^{2}d)$

### 计算attention的时候为何选择点乘而不是加法？两者计算复杂度和效果上有什么区别？

### 在计算attention score的时候如何对padding做mask操作？

对需要mask的位置设为负无穷，再对attention score进行相加

## Layer Norm

```python
# features: (bsz, max_len, hidden_dim)
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
	super(LayerNorm, self).__init__()
	self.a_2 = nn.Parameter(torch.ones(features))
	self.b_2 = nn.Parameter(torch.zeros(features))
	self.eps = eps
	
    def forward(self, x):
	# 就是在统计每个样本所有维度的值，求均值和方差，所以就是在hidden dim上操作
	# 相当于变成[bsz*max_len, hidden_dim], 然后再转回来, 保持是三维
	mean = x.mean(-1, keepdim=True) # mean: [bsz, max_len, 1]
	std = x.std(-1, keepdim=True) # std: [bsz, max_len, 1]
        # 注意这里也在最后一个维度发生了广播
	return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
```

### add&norm的作用？

利用的resnet的残差连接， 一是解决梯度消失的问题，二是解决权重矩阵的退化问题。

对权重矩阵的退化问题补充一点解释：随着网络深度增加，模型会产生退化现象。它不是由过拟合产生的，而是由冗余的网络层学习了不是恒等映射的参数造成的。利用残差连接，在前向传播时，输入信号可以从任意低层直接传播到高层。由于包含了一个天然的恒等映射，一定程度上可以解决网络退化问题。

### layer norm和batch norm的区别？为什么cv常用bn，nlp常用ln？

Norm的作用是平滑了Loss， layer norm在多头注意力层和激活函数层之间。CV使用BN是认为channel维度的信息对cv方面有重要意义，如果对channel维度也归一化会造成不同通道信息一定的损失。而同理nlp领域认为句子长度不一致，并且各个batch的信息没什么关系，因此只考虑句子内信息的归一化，也就是LN。

### 简答讲一下BatchNorm技术，以及它的优缺点。

BN是对每一批的数据在进入激活函数前进行归一化，可以提高收敛速度，防止过拟合，防止梯度消失，增加网络对数据的敏感度。

## Position Embedding

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        """
        :param d_model: pe编码维度，一般与word embedding相同，方便相加
        :param dropout: dorp out
        :param max_len: 语料库中最长句子的长度，即word embedding中的L
        """
        super(PositionalEncoding, self).__init__()
        # 定义drop out
        self.dropout = nn.Dropout(p=dropout)
        # 计算pe编码
        pe = torch.zeros(max_len, d_model) # 建立空表，每行代表一个词的位置，每列代表一个编码位
        position = torch.arange(0, max_len).unsqueeze(1) # 建个arrange表示词的位置以便公式计算，size=(max_len,1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *    # 计算公式中10000**（2i/d_model)
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 计算偶数维度的pe值
        pe[:, 1::2] = torch.cos(position * div_term)  # 计算奇数维度的pe值
        pe = pe.unsqueeze(0)  # size=(1, L, d_model)，为了后续与word_embedding相加,意为batch维度下的操作相同
        self.register_buffer('pe', pe)  # pe值是不参加训练的

    def forward(self, x):
        # 输入的最终编码 = word_embedding + positional_embedding
        x = x + Variable(self.pe[:, :x.size(1)],requires_grad=False) #size = [batch, L, d_model]
        return self.dropout(x) # size = [batch, L, d_model]
```

### 解释一下transformer的位置编码

因为self-attention是位置无关的，无论句子的顺序是什么样的，通过self-attention计算的token的hidden embedding都是一样的，这显然不符合人类的思维。因此要有一个办法能够在模型中表达出一个token的位置信息，transformer使用了固定的positional encoding来表示token在句子中的绝对位置信息。作者们使用了不同频率的正弦和余弦函数来作为位置编码。

参考：https://www.zhihu.com/question/347678607

https://zhuanlan.zhihu.com/p/360539748

### 为什么Transformer的位置编码可以反应出相对位置

如何理解Transformer论文中的positional encoding，和三角函数有什么关系？ - 猛猿的回答 - 知乎 https://www.zhihu.com/question/347678607/answer/2301693596

**两个位置编码的点积(dot product)仅取决于[偏移量](https://www.zhihu.com/search?q=偏移量&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2301693596})** △t **，也即两个位置编码的点积可以反应出两个位置编码间的距离。** （运用和角公式）

### 其它位置编码技术的优缺点？

1. 整型值编码：[1, 2,..., n]，直观。缺点：模型在测试时可能遇到比训练数据更长的序列，不利于泛化。随着序列长度的增加，位置值会越来越大。
2. 用[0,1]值标记位置。缺点：序列长度不一致的时候，token之间的相对距离不一样。
3. 二进制编码。可以生成一个和d_model维度一致的二进制数。缺点：这样编码出来的位置向量，处在一个离散的空间中，不同位置间的变化是不连续的
4. 用周期函数来表示位置。把周期函数的波长拉长来表示位置，避免首尾的距离太相近
5. 用sin和cos交替来表示位置。不仅能表示一个token的绝对位置，还可以表示一个token的相对位置

### 你还了解什么关于位置编码的技术？各自的优缺点是什么？

相对位置编码（RPE）

1.在计算attention score和weighted value时各加入一个可训练的表示相对位置的参数。

2.在生成多头注意力时，把对key来说将绝对位置转换为相对query的位置

3.复数域函数，已知一个词在某个位置的词向量表示，可以计算出它在任何位置的词向量表示。前两个方法是词向量+位置编码，属于亡羊补牢，复数域是生成词向量的时候即生成对应的位置信息。

### Transformer在哪里做了权重共享，为什么可以做权重共享？

Transformer在两个地方进行了权重共享：

**（1）**Encoder和Decoder间的Embedding层权重共享；

**（2）**Decoder中Embedding层和FC层权重共享。

**对于（1）**，《Attention is all you need》中Transformer被应用在机器翻译任务中，源语言和目标语言是不一样的，但它们可以共用一张大词表，对于两种语言中共同出现的词（比如：数字，标点等等）可以得到更好的表示，而且对于Encoder和Decoder，**嵌入时都只有对应语言的embedding会被激活**，因此是可以共用一张词表做权重共享的。论文中，Transformer词表用了bpe来处理，所以最小的单元是subword。英语和德语同属日耳曼语族，有很多相同的subword，可以共享类似的语义。而像中英这样相差较大的语系，语义共享作用可能不会很大。但是，共用词表会使得词表数量增大，增加softmax的计算时间，因此实际使用中是否共享可能要根据情况权衡。

**对于（2）**，Embedding层可以说是通过onehot去取到对应的embedding向量，FC层可以说是相反的，通过向量（定义为 x）去得到它可能是某个词的softmax概率，取概率最大（贪婪情况下）的作为预测值。那哪一个会是概率最大的呢？在FC层的每一行量级相同的前提下，理论上和 x 相同的那一行对应的点积和softmax概率会是最大的（可类比本文问题1）。因此，Embedding层和FC层权重共享，Embedding层中和向量 x 最接近的那一行对应的词，会获得更大的预测概率。实际上，Decoder中的**Embedding层和FC层有点像互为逆过程**。通过这样的权重共享可以减少参数的数量，加快收敛。



# 预训练模型

发展史：ELMO→GPT→BERT→GPT-2→Prompt

ELMO：基于双向语言模型和双向LSTM预训练

GPT：放弃LSTM，改用Transformer decoder，基于单向语言模型预训练

BERT：基于Transformer encoder和双向语言模型，两个任务：masked LM和next sentence prediction

GPT-2：更大规模的GPT

Prompt：改变下游任务的形态使其适配预训练模型

# BERT

## BERT基本结构

## BERT的两个训练任务

1. 随机Mask
2. 预测两句话是否互为上下文

BERT的两个预训练任务（MLM和NSP），一起还是分开，分开的话谁先谁后

简单描述为：在一句话中mask掉几个单词然后对mask掉的单词做预测、判断两句话是否为上下文的关系，而这两个训练任务是同时进行的，也就意味着训练语料的形式大约为：

[CLS]谷歌和[MASK]都是不存在的。[SEP]同时，[MASK]也是不存在的。[SEP]

https://zhuanlan.zhihu.com/p/74090249

## BERT变种

### ALBERT

减少模型参数： 

1. embedding size从768变为256
2. encoder中每层参数共享，主要是attention
3. 新任务SOP（sentence order prediction 句子语序预测）。正例：一篇文章的连续语句，负例却是两句掉换位置
4. 移除dropout

### RoBERTa

1. 没有了NSP任务
2. 更大的预训练语料
3. 对每条语料copy10份，每一份都是不同的mask
4. Word-piece切分tokenizer

### XL-Net

随机打散序列，乱序mask。通过调整position embedding

### Transformer-XL

主要解决输入长度限制的问题。依旧会将长序列分成n段。

创新点：将i-1段得到的信息加入到i段的训练中，但bp时第i段不回传到i-1段。（类似于缓存机制）

### fastBERT

减少BERT的decoder层数。先finetune，蒸馏时每一层decoder加上分类网络，计算其与最后一层的KL/JS散度。通过分类的不确定度（Uncertainty）是否达到阈值来判断是否再需要后续的decoder层。

### 简述ELMO和BERT的区别，以及各自的优缺点

ELMo是用两个方向的语言模型来训练bi-LSTM,其瓶颈在于能力受限而无法使模型变得更深。BERT使用了一种反其道而行之的套路，设置一个比语言模型（Language Model，简称LM）更简单的任务来做预训练，并且使用基于Transformer的Encoder来进行预训练从而使得模型变深。

### 简述GPT和BERT的区别，以及各自的优缺点

## BERT和Transformer的位置编码区别

BERT的位置编码是学习出来的，Transformer是通过正弦函数生成的。

原生的Transformer中使用的是正弦位置编码（Sinusoidal Position Encoding），是绝对位置的函数式编码。由于Transformer中为self-attention，这种正余弦函数由于点乘操作，会有相对位置信息存在，但是没有方向性，且通过权重矩阵的映射之后，这种信息可能消失。

BERT中使用的是学习位置嵌入（learned position embedding），是绝对位置的参数式编码，且和相应位置上的词向量进行相加而不是拼接。

## BERT embedding layer有三个嵌入层的含义？

Token-embedding：将单词转换为固定维的向量表示形式，在BERT-base中，每个单词都表示为一个768维的向量。

Segment-embedding：BERT在解决双句分类任务（如判断两段文本在语义上是否相似）时是直接把这两段文本拼接起来输入到模型中，模型是通过segment-embedding区分这两段文本。对于两个句子，第一个句子的segment-embedding部分全是0，第二个句子的segment-embedding部分全是1。

Position-embedding：BERT使用transformer编码器，通过self-attention机制学习句子的表征，self-attention不关注token的位置信息，所以为了能让transformer学习到token的位置信息，在输入时增加了position-embedding。



# GPT

# GPT-2

# Prompt

https://juejin.cn/post/7061997371785216037

## NLP范式演化历程

基于非神经网络的全监督学习→基于神经网络的全监督学习→预训练+微调→prompt

**全监督学习（非神经网络）：** 仅在目标任务的输入输出样本数据集上训练特定任务模型，严重依赖特征工程。

**全监督学习（神经网络）：** 使得特征学习与模型训练相结合，于是研究重点转向了架构工程，即通过设计一个网络架构（如CNN，RNN，Transformer）能够学习数据特征。

**Pre-train，Fine-tune：** 先在大数据集上预训练，再根据特定任务对模型进行微调，以适应于不同的下游任务。在这种范式下，研究重点转向了目标工程，设计在预训练和微调阶段使用的训练目标（损失函数）。

**Pre-train，Prompt，Predict：** 无需要fine-tune，让预训练模型直接适应下游任务。方便省事，不需要每个任务每套参数，突破数据约束。

## Prompt是什么

将下游任务的输入输出形式改造成预训练任务中的形式，即 MLM (Masked Language Model) 的形式。

**原任务：**

Input: `I love this movie.`

Output: `++ (very positive)`

**改造后：**

Prefix prompt 版（prompt 槽在文本末尾，适合生成任务或自回归 LM，如 GPT-3）：
 Input: `I love this movie. Overall, the movie is [Z].`

Cloze prompt 版（prompt 槽在文本中间或结尾，适合 MLM 任务，如 BERT）：
 Input: `I love this movie. Overall, it was a [Z] movie.`

Output: `[Z] = ‘good’`



![](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/3622bcb6076f48ee9bc9dbedaf96b985~tplv-k3u1fbpfcp-zoom-in-crop-mark:3024:0:0:0.awebp?)

之前的预训练+微调是让预训练模型去适应下游任务，而 Prompt 则是调整下游任务来适应预训练模型。

![](https://ucc.alicdn.com/pic/developer-ecology/6ec32c8401bd477bb777b34d5b351af4.png)

生成任务的常用Prompt如上图所示。

## Why Prompt Works？

比起微调从零开始学习一个分类器（举例），建立预训练模型输出与分类结果之间的对应，Prompt 的任务形式与预训练相同，直接可以从输入中获取更多的语义信息，因此即便是少量数据甚至是 zero-shot 也可能取得不错的效果。

## Prompt 的优点

如上所述，prompt 的引入使得预训练模型提取的特征更自然地用于下游任务的预测，特征质量更高。不需要为下游任务新增一个分类器，因为任务形式与预训练模型本身相适应；也不需要从零开始训练本来要加的这个分类器。只需要建立一个简单的映射，将 prompt 范式的输出再转变成下游任务需要的输出形式。在少样本甚至零样本场景下表现优秀。

## How Prompt

### 如何构建 Prompt 的 pipeline

- Prompt Addition：在输入中添加 Prompt；
- Answer Search：根据改造后的输入预测[Z]；
- Answer Mapping：把预测的结果转变成下游任务所需要的形式。

### 如何设计自己的 Prompt 模型

- **预训练模型**的选择；

- Prompt Engineering

  ：选择合适的 Prompt，包括两方面：

  - prefix prompt 还是 cloze prompt？
  - 手动设计还是自动构建（搜索、优化、生成等）？

- **Answer Engineering**：选择合适的方法将预测结果映射回下游任务需要的输出形式；

- **Multi-prompt**：设计多个 prompt 以获取更好的效果（集成学习、数据增强等）；

- 训练策略

  ：Prompt 模型可能会含有除了 LM 模型以外的 Prompt 参数，训练策略需要考虑的包括：

  - 有没有额外的 Prompt Params？
  - 是否更新这些 Prompt 参数？


