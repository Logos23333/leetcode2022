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

SGD：更新比较频繁，会造成 cost function 有严重的震荡，最终停留在Local Minima或者Saddle Point处。

SGD with Momentum： 提出了动量的概念，为了跳出局部最优。使用了一阶动量， 在梯度下降的过程中加入了惯性，使得梯度方向不变的维度上速度变快，梯度方向有所改变的维度上的更新速度变慢，这样就可以加快收敛并减小震荡。一句话：SGDM不仅考虑当前梯度，也考虑到了上个时间的梯度。

AdaGrad：  二阶动量为该维度上迄今为止所有梯度值的平方和。缺点：学习率会越来越小

RMSProp： 窗口滑动加权平均值计算二阶动量。

Adam：一阶动量+窗口滑动二阶动量

## boosting

boosting是一族可以将弱学习器提升为强学习器的算法，先从初始训练集中训练出一个基学习器，再根据基学习器的表现对训练样本分布进行调整，使得先前基学习器做错的训练样本在后续得到更多关注，然后基于调整后的样本分布来训练下一个基分类器，重复进行，直到基学习器数目达到实现指定的值T。最后加权求和。boosting算法主要关注降低偏差。

## bagging

给定包含m个样本的数据集，放回的进行m次随机采样，会得到一个含m个样本的采样集。按照这样的采样方式，采样T次，得到T个不同的采样集，基于每个采样集训练出一个基学习器，对分类任务采用简单投票法，对回归任务使用简单平均法。bagging算法主要关注降低方差。

## 随机森林

随机森林是Bagging算法的一个变种，它在以决策树为基学习器构建Bagging集成的基础上，进一步在决策树的训练过程引入了随机属性选择。跟dropput有着异曲同工之妙。

# transformer

## transformer的结构

<img src="https://pic1.zhimg.com/80/v2-4b53b731a961ee467928619d14a5fd44_1440w.jpg" style="zoom: 50%;" />

### add&norm的作用？

利用的resnet的残差连接， 一是解决梯度消失的问题，二是解决权重矩阵的退化问题。

### layer norm和batch norm的区别？为什么cv常用bn，nlp常用ln？

Norm的作用是平滑了Loss， layer norm在多头注意力层和激活函数层之间。CV使用BN是认为channel维度的信息对cv方面有重要意义，如果对channel维度也归一化会造成不同通道信息一定的损失。而同理nlp领域认为句子长度不一致，并且各个batch的信息没什么关系，因此只考虑句子内信息的归一化，也就是LN。

### 简答讲一下BatchNorm技术，以及它的优缺点。

BN是对每一批的数据在进入激活函数前进行归一化，可以提高收敛速度，防止过拟合，防止梯度消失，增加网络对数据的敏感度。

### 为什么transformer的decoder有masked multi-head attention，有什么区别？意义？

是因为在decode的时候预测第i个词的时候遮住i后面的词。

### transformer除了decoder有mask，还有哪里做了mask操作？为什么？

整个Transformer中包含三种类型的attention,且目的并不相同。

- Encoder的self-attention，考虑到batch的并行化，通常会进行padding，因此会对序列中mask=0的token进行mask后在进行attention score的softmax归一化。
- Decoder中的self-attention，为了避免预测时后续tokens的影所以必须令后续tokens的mask=0，其具体做法为构造一个三角矩阵。
- Decoder中的encode-decoder attention，涉及到decoder中当前token与整个encoder的sequence的计算，所以encoder仍然需要考虑mask。

综上，无论对于哪个类型的attention，在进行sotmax归一化前，都需要考虑mask操作。

### 为什么要multi-head?

多头可以使参数矩阵形成多个子空间，矩阵整体的size不变，只是改变了每个head对应的维度大小，这样做使矩阵对多方面信息进行学习，但是计算量和单个head差不多

### Decoder阶段的mha和encoder的mha有什么区别？

Decoder有两层mha，encoder有一层mha，Decoder的第二层mha是为了转化输入与输出句长，Decoder的请求q与键k和数值v的倒数第二个维度可以不一样，但是encoder的qkv维度一样。

### bert的mask为何不学习transformer在attention处进行屏蔽score的技巧？

BERT和transformer的目标不一致，bert是语言的预训练模型，需要充分考虑上下文的关系，而transformer主要考虑句子中第i个元素与前i-1个元素的关系。

### 解释一下transformer的位置编码

因为self-attention是位置无关的，无论句子的顺序是什么样的，通过self-attention计算的token的hidden embedding都是一样的，这显然不符合人类的思维。因此要有一个办法能够在模型中表达出一个token的位置信息，transformer使用了固定的positional encoding来表示token在句子中的绝对位置信息。作者们使用了不同频率的正弦和余弦函数来作为位置编码。

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

## self-attention

$softmax(\frac{QK^{T}}{ \sqrt d_{k}})\cdot V$

### 为什么使用不同的Q和K？

是为了打破对称性，参考其中“如果令Q=K，那么得到的模型大概率会得到一个类似单位矩阵的attention矩阵，**这样self-attention就退化成一个point-wise线性映射**。这样至少是违反了设计的初衷。”

知乎 https://www.zhihu.com/question/319339652https://www.bilibili.com/read/cv4902832

### 为什么要除以delta dk，dk的含义是？

向量的点积结果会很大，将softmax函数的梯度就会很小，scaled会缓解这种现象。在输入的数量级很大时，softmax的梯度会消失为0， 造成参数更新困难。dk是Q\*K的方差，方差越大说明点积的数量级越大，在除以根号dk后方差变为1，** **也就有效地控制了前面提到的梯度消失的问题。

### 怎么估计self-attention的时间复杂度？

$Q\cdot K$的复杂度为$O(n^{2}d)$，$Q$和$K$的维度为$n*h$

对$n*n$做softmax，复杂度为$O(n^2)$

$(n*n) \cdot (n*d)$的复杂度为$O(n^2d)$

### 计算attention的时候为何选择点乘而不是加法？两者计算复杂度和效果上有什么区别？

### 在计算attention score的时候如何对padding做mask操作？

对需要mask的位置设为负无穷，再对attention score进行相加

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

减少BERT的decoder层数。先finetune，蒸馏时每一层decoder加上分类网络，计算其与最后一层的KL/JS散度。通过分类的不确定度（Uncertainty）是否达到阈值来判断是否再需要后续的decoder层