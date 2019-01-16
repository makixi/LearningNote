# Decision Tree 

---

### 1.Decision Tree Hypothesis
aggregation的核心就是将许多可供选择使用的比较好的hypothesis融合起来，利用集体的智慧组合成G，使其得到更好的机器学习预测模型。<br>

![1](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/9_1.png?raw=true)<br>
aggregation type有三种：uniform，non-uniform，conditional。<br>
它有两种情况，一种是所有的g是已知的，即blending。对应的三种类型分别是voting/averaging，linear和stacking。<br>
另外一种情况是所有g未知，只能通过手上的资料重构g，即learning。其中uniform和non-uniform分别对应的是Bagging和AdaBoost算法，而conditional对应的就是Decision Tree算法。<br>

***

决策树的整个流程类似一个树状结构。<br>
![2](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/9_2.png?raw=true)<br>

图中每个条件和选择都决定了最终的结果，Y or N。<br>
圆圈表示树的叶子，即最终的决定。<br>
<br>
这种树状结构对应到一个hypothesis G(x)中，G(x)的表达式为：

$$G(x)=\sum_{t=1}^Tq_t(x)\cdot g_t(x)$$

G(x)由许多$g_t(x)$组成，即aggregation的做法。<br>
每个$g_t(x)$就代表上图中的圆圈（树的叶子）。这里的$g_t(x)$是常数，因为是处理简单的classification问题。我们把这些$g_t(x)$称为base hypothesis。<br>
$q_t(x)$表示每个$g_t(x)$成立的条件，代表上图中橘色箭头的部分。不同的$g_t(x)$对应于不同的$q_t(x)$，即从树的根部到顶端叶子的路径不同。<br>
base hypothesis和conditions就构成了整个G(x)的形式，从根部到顶端所有的叶子都安全映射到上述公式上去了。<br>

![3](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/9_3.png?raw=true)<br>

利用条件分支的思想，将整体G(x)分成若干个$G_c(x)$，也就是把整个大树分成若干个小树，如下所示：

$$G(x)=\sum_{c=1}^C[b(x)=c]\cdot G_c(x)$$

G(x)表示完整的大树，即full-tree hypothesis，b(x)表示每个分支条件，即branching criteria，$G_c(x)$表示第c个分支下的子树，即sub-tree。<br>
这种结构被称为递归型的数据结构，即将大树分割成不同的小树，再将小树继续分割成更小的子树。<br>
决策树可以分为两部分：root和sub-trees。<br>

![4](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/9_4.png?raw=true)<br>

***

优点  | 缺点
---- | ---
模型直观，便于理解，应用广泛  | 缺少足够的理论支持
算法简单，容易实现  | 如何选择合适的树结构对初学者来说比较困惑
训练和预测时，效率较高  | 决策树代表性的演算法比较少

![5](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/9_5.png?raw=true)<br>

---

### 2.Decision Tree Algorithm
用递归形式将decision tree表示出来，它的基本的算法可以写成：<br>
![6](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/9_6.png?raw=true)<br>
<br>

Basic Decision Tree Algorithm的流程可以分成四个部分：<br>
1.学习设定划分不同分支的标准和条件是什么<br>
2.将整体数据集D根据分支个数C和条件，划为不同分支下的子集Dc<br>
3.对每个分支下的Dc进行训练，得到相应的机器学习模型Gc<br>
4.将所有分支下的Gc合并到一起，组成大矩G(x)<br>
<br>
当满足递归的终止条件之后，将会返回基本的hypothesis $g_t(x)$。<br>
![7](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/9_7.png?raw=true)<br>
决策树的基本演算法包含了四个选择：<br>
1.分支个数（number of branches）<br>
2.分支条件（branching criteria）<br>
3.终止条件（termination criteria）<br>
4.基本算法（base hypothesis）<br>

***

eg:一种常用的决策树模型算法，叫做Classification and Regression Tree(C&RT)。<br>
C&RT算法有两个简单的设定:<br>
1.分支的个数C=2，即二叉树（binary tree）的数据结构<br>
2.每个分支最后的$g_t(x)$（数的叶子）是一个常数<br>
<br>
按照最小化$E_{in}$的目标，对于binary/multiclass classification(0/1 error)问题，看正类和负类哪个更多，$g_t(x)$取所占比例最多的那一类$y_n$<br>
对于regression(squared error)问题，$g_t(x)$则取所有$y_n$的平均值。<br>
![8](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/9_8.png?raw=true)<br>

***

对于决策树的基本演算法流程，C&RT还有一些简单的设定。<br>
首先，C&RT分支个数C=2，一般采用decision stump的方法进行数据切割。也就是每次在一个维度上，只对一个特征feature将数据一分为二，左子树和右子树，分别代表不同的类别。<br>
C&RT中使用纯净度purifying这个概念来选择最好的decision stump。<br>
purifying的核心思想就是每次切割都尽可能让左子树和右子树中同类样本占得比例最大或者$y_n$都很接近（regression），即错误率最小。<br>
![9](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/9_9.png?raw=true)<br>

***

根据C&RT中purifying的思想，我们得到选择合适的分支条件b(x)。<br>
最好的decision stump重点包含两个方面：<br>
1.purifying越大越好，而与purifying相反的概念是impurity，则impurity越小越好；<br>
2.左右分支纯净度所占的权重，权重大小由该分支的数据量决定，分支包含的样本个数越多，则所占权重越大，分支包含的样本个数越少，则所占权重越小。上式中的$|D_c\ with\ h|$代表了分支c所占的权重。这里b(x)类似于error function（这也是为什么使用impurity代替purifying的原因），选择最好的decision stump，让所有分支的不纯度最小化，使b(x)越小越好。

对于regression问题，它的impurity可表示为：

$$impurity(D)=\frac1N\sum_{n=1}^N(y_n-\overline{y})^2$$

其中，$\overline{y}$表示对应分支下所有$y_n$的均值。

对应classification问题，它的impurity可表示为：

$$impurity(D)=\frac1N\sum_{n=1}^N[y_n\neq y^*]$$

其中，$y^*$表示对应分支下所占比例最大的那一类。<br>
![10](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/9_10.png?raw=true)<br>

以上这些impurity是基于原来的regression error和classification error直接推导的。<br>
进一步来看classification的impurity functions，如果某分支条件下，让其中一个分支纯度最大，那么就选择对应的decision stump，即得到的classification error为：

$$1-max_{1\leq k\leq K}\frac{\sum_{n=1}^N[y_n=k]}{N}$$

其中，K为分支个数。

上面这个式子只考虑纯度最大的那个分支，更好的做法是将所有分支的纯度都考虑并计算在内，用基尼指数（Gini index）表示：

$$1-\sum_{k=1}^K(\frac{\sum_{n=1}^N[y_n=k]}{N})^2$$

Gini index的优点是将所有的class在数据集中的分布状况和所占比例全都考虑了，这样让decision stump的选择更加准确。<br>
![11](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/9_11.png?raw=true)<br>
对于决策树C&RT算法，通常来说，Gini index更适合求解classification问题，而regression error更适合求解regression问题。

C&RT算法迭代终止条件有两种情况:<br>
1.当前各个分支下包含的所有样本$y_n$都是同类的，即不纯度impurity为0，表示该分支已经达到了最佳分类程度<br>
2.该特征下所有的$x_n$相同，无法对其进行区分，表示没有decision stumps。<br>
![12](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/9_12.png?raw=true)<br>
C&RT算法遇到迭代终止条件后就成为完全长成树（fully-grown tree）。<br>
它每次分支为二，是二叉树结构，采用purify来选择最佳的decision stump来划分，最终得到的叶子（$g_t(x)$）是常数。

---

### 3.Decision Tree Heuristics in C&RT
C&RT算法的基本流程：<br>
![13](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/9_13.png?raw=true)<br>

C&RT算法在处理binary classification和regression问题时非常简单实用，而且，处理muti-class classification问题也十分容易。

为了避免overfit，我们需要在C&RT算法中引入正则化，来控制整个模型的复杂度。

考虑到避免模型过于复杂的方法是减少叶子（$g_t(x)$）的数量，那么可以令regularizer就为决策树中叶子的总数，记为$\Omega(G)$。<br>
正则化的目的是尽可能减少$\Omega(G)$的值。<br>
regularized decision tree的形式就可以表示成：

$$argmin_{(all\ possible\ G)}\ E_{in}(G)+\lambda\Omega(G)$$

我们把这种regularized decision tree称为pruned decision tree。<br>
通过regularization来修剪决策树，从而达到避免过拟合的效果。

那么如何确定修剪多少叶子，修剪哪些叶子呢？<br>
假设由C&RT算法得到一棵完全长成树（fully-grown tree），总共10片叶子。首先分别减去其中一片叶子，剩下9片，将这10种情况比较，取$E_{in}$最小的那个模型；然后再从9片叶子的模型中分别减去一片，剩下8片，将这9种情况比较，取$E_{in}$最小的那个模型。以此类推，继续修建叶子。这样，最终得到包含不同叶子的几种模型，将这几个使用regularized decision tree的error function来进行选择，确定包含几片叶子的模型误差最小，就选择该模型。<br>
参数$\lambda$可以通过validation来确定最佳值。<br>
![14](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/9_14.png?raw=true)<br>

***

对于numerical features，我们直接使用decision stump进行数值切割；<br>
而对于categorical features，我们仍然可以使用decision subset，对不同类别进行“左”和“右”，即是与不是（0和1）的划分。<br>
numerical features和categorical features的具体区别如下图所示：<br>
![15](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/9_15.png?raw=true)<br>

***

当某些特征缺失的时候，没有办法进行切割和分支选择。<br>
一种常用的方法就是surrogate branch，即寻找与该特征相似的替代feature。如果替代的feature与原feature切割的方式和结果是类似的，那么就表明二者是相似的，就把该替代的feature也存储下来。当预测时遇到原feature缺失的情况，就用替代feature进行分支判断和选择。<br>
![16](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/9_16.png?raw=true)<br>

---

### 4. Decision Tree in Action
下图二维平面上分布着许多正负样本，我们使用C&RT算法来对其进行决策树的分类。<br>
![17](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/9_17.png?raw=true)<br>
<br>
第一步：<br>
![18](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/9_18.png?raw=true)<br>
<br>
第二步：<br>
![19](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/9_19.png?raw=true)<br>
<br>
第三步：<br>
![20](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/9_20.png?raw=true)<br>
<br>
第四步：<br>
![21](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/9_21.png?raw=true)<br>
<br>
在进行第四步切割之后，我们发现每个分支都已经非常纯净了，没有办法继续往下切割。<br>
此时已经满足了迭代终止条件，这时候就可以回传base hypothesis，构成sub tree，然后每个sub tree再往上整合形成tree，最后形成完全决策树。<br>
如果将边界添加上去，可得到下图：<br>
![22](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/9_22.png?raw=true)<br>
<br>
得到C&RT算法的切割方式之后，我们与AdaBoost-Stump算法进行比较：<br>
![23](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/9_23.png?raw=true)<br>

AdaBoost-Stump算法的切割线是横跨整个平面的；而C&RT算法的切割线是基于某个条件的，所以一般不会横跨整个平面。<br>
虽然C&RT和AdaBoost-Stump都采用decision stump方式进行切割，但是二者在细节上还是有所区别。<br>
<br>
再看一个数据集分布比较复杂的例子，C&RT和AdaBoost-Stump的切割方式对比效果如下图所示：<br>

![24](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/9_24.png?raw=true)<br>

***

通常来说，由于C&RT是基于条件进行切割的，所以C&RT比AdaBoost-Stump分类切割更有效率。<br>
总结一下，C&RT决策树的特点：<br>
![25](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/9_25.png?raw=true)<br>

