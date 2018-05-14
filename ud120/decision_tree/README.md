# 决策树
@(机器学习)
- 决策树引例：今天适合冲浪吗🏄
  ![|500x0](https://ws1.sinaimg.cn/large/006tNc79gy1fr9ztvyrcij316y0rctny.jpg)
- 切割3次
  ![|500x0](https://ws2.sinaimg.cn/large/006tNc79gy1fr9zzzkhzuj318s0seaq3.jpg)
- 所谓决策树：就是计算机根据算法找到决策边界
- 参数
```python
class sklearn.tree.DecisionTreeClassifier(criterion=’gini’, splitter=’best’, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)
```
> criterion：标准
> splitter：分隔方式
> max_depth：最大深度
> min_samples_split：可分隔样本数量下限
> min_samples_leaf：叶子节点的样本数量下限
> ![|500x0](https://ws4.sinaimg.cn/large/006tNc79gy1fra1fuk6qrj31dg0iuwqw.jpg)
> ```min_samples_split```默认是2，由上图只有1个叶子节点不会在继续分割了。这个值设置太小，将会形成较为**复杂**的决策树，导致**过拟合**

- Entropy 熵
  熵——决定了决策树在何处分隔数据
  找到变量划分点，从而产生尽可能单一的子集，递归进行这一过程

  
>信息量$-log_2p(x)$的大小跟事情不确定性的变化有关。

>例如我们讨论太阳从哪升起。本来就只有一个结果，我们早就知道，那么无论谁传递任何信息都是没有信息量的。

>当可能结果数量比较大时，我们得到的新信息才有潜力拥有大信息量。

>在数学上，信息熵其实是信息量的期望。

>信息熵是信息的不确定性（Uncertainty）的度量，不确定性越大，信息熵越大。

$$Entropy = -\sum_ip(x)log_2p(x)$$
$p(x)$是第x类中的样本占总样本数的比例
如果Entropy越大，说明数据越不纯，蕴含的信息量越大

所有样本属于同一类，$log_21=0$，熵为0
样本均匀分布在所有的类中，熵为1

- 信息增益（Information Gain）
  ![|500x0](https://ws1.sinaimg.cn/large/006tNc79gy1franaa62soj31kw0f4n71.jpg)

决策树算法：**最大化信息增益！**
![|400x0](https://ws2.sinaimg.cn/large/006tNc79gy1frao0z520fj30ne0fcwln.jpg)

![|400x0](https://ws2.sinaimg.cn/large/006tNc79gy1franme1cjhj30he0ayq4n.jpg)

1. 计算某个节点的熵

$$Entropy = -\sum_ip(x)log_2p(x)$$

	>>> import math
	>>> -2.0/3 * math.log(2.0/3 , 2) - 1.0/3 * math.log(1.0/3 , 2)
	>>> 0.9182958340544896
	# 或者是使用scipy.stats
	import scipy.stats
	print scipy.stats.entropy([2,1],base=2)

![|400x0](https://ws3.sinaimg.cn/large/006tNc79gy1franopp5g8j30oa07owgl.jpg)

2. 计算信息增益(以Grade划分)
  ![|500x0](https://ws4.sinaimg.cn/large/006tNc79gy1frao2i8cu3j31he0p6000.jpg)
  根据(Grade)划分的信息增益 = 1 - 3/4 * 0.9184 = 0.3112
  根据(bumpiness)划分的信息增益为0
  根据(speed)划分的信息增益为1
  ![|500x0](https://ws1.sinaimg.cn/large/006tNc79gy1fraohwd8klj31gu0rmqtt.jpg)