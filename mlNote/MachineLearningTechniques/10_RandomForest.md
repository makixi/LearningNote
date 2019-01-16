# Random Forest

---

### 1.Random Forest Algorithm
Bagging是通过bootstrap的方式，从原始的数据集D中得到新的$\hat{D}$；然后再使用一些base algorithm对每个$\hat{D}$都得到相应的$g_t$；最后将所有的$g_t$通过投票uniform的形式组合成一个G，G即为我们最终得到的模型。<br>
Decision Tree是通过递归形式，利用分支条件，将原始数据集D切割成一个个子树结构，长成一棵完整的树形结构。Decision Tree最终得到的G(x)是由相应的分支条件b(x)和分支树$G_c(x)$递归组成。<br>
![1](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/10_1.png?raw=true)<br>

***

Bagging和Decison Tree算法各自有一个很重要的特点。<br>
Bagging具有减少不同$g_t$的方差variance的特点。这是因为Bagging采用投票的形式，将所有$g_t$uniform结合起来，起到了求平均的作用，从而降低variance。<br>
Decision Tree具有增大不同$g_t$的方差variance的特点。这是因为Decision Tree每次切割的方式不同，而且分支包含的样本数在逐渐减少，所以它对不同的资料D会比较敏感一些，从而不同的D会得到比较大的variance。

所以说，Bagging能减小variance，而Decision Tree能增大variance。<br>
那么可以使用Bagging的方式把众多的Decision Tree进行uniform结合起来。<br>
这种算法就叫做随机森林（Random Forest），它将完全长成的C&RT决策树通过bagging的形式结合起来，最终得到一个庞大的决策模型。<br>
![2](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/10_2.png?raw=true)<br>

***

Random Forest算法流程图如下所示：<br>
![3](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/10_3.png?raw=true)<br>
Random Forest算法的优点主要有三个：<br>
1.不同决策树可以由不同主机并行训练生成，效率很高<br>
2.随机森林算法继承了C&RT的优点<br>
3.将所有的决策树通过bagging的形式结合起来，避免了单个决策树造成过拟合的问题。<br>
![4](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/10_4.png?raw=true)<br>
以上是基本的Random Forest算法。<br>
Bagging中，通过bootstrap的方法得到不同于D的D'，使用这些**随机抽取的资料**得到不同的$g_t$。除了随机抽取资料获得不同$g_t$的方式之外，还有另外一种方法，就是**随机抽取一部分特征**。例如，原来有100个特征，现在只从中随机选取30个来构成决策树，那么每一轮得到的树都由不同的30个特征构成，每棵树都不一样。<br>
假设原来样本维度是d，则只选择其中的d'（d'小于d）个维度来建立决策树结构。这类似是一种从d维到d'维的特征转换，相当于是从高维到低维的投影，也就是说d'维z空间其实就是d维x空间的一个随机子空间（subspace）。通常情况下，d'远小于d，从而保证算法更有效率。<br>
Random Forest算法的作者建议在构建C&RT每个分支b(x)的时候，都可以重新选择子特征来训练，从而得到更具有多样性的决策树。<br>
![5](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/10_5.png?raw=true)<br>
所以说，这种增强的Random Forest算法增加了random-subspace。<br>
![6](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/10_6.png?raw=true)<br>
上面我们讲的是随机抽取特征，除此之外，还可以将现有的特征x，通过数组p进行线性组合，来保持多样性：

$$\phi_i(x)=p_i^Tx$$

这种方法使每次分支得到的不再是单一的子特征集合，而是子特征的线性组合（权重不为1）。<br>
好比在二维平面上不止得到水平线和垂直线，也能得到各种斜线。这种做法使子特征选择更加多样性。<br>
不同分支i下的$p_i$是不同的，而且向量$p_i$中大部分元素为零，因为我们选择的只是一部分特征，这是一种低维映射。
![7](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/10_7.png?raw=true)<br>
所以，这里的Random Forest算法又有增强，由原来的random-subspace变成了random-combination。<br>
这里的random-combination类似于perceptron模型。<br>
![8](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/10_8.png?raw=true)<br>

---

### 2.Out-Of-Bag Estimate
通过bootstrap得到新的样本集D'，再由D'训练不同的$g_t$。<br>
我们知道D'中包含了原样本集D中的一些样本，但也有些样本没有涵盖进去。<br>
如下表所示，不同的$g_t$下，红色的表示在$\hat D_t$中没有这些样本。<br>
每个$g_t$中，红色表示的样本被称为out-of-bag(OOB) example。<br>
![9](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/10_9.png?raw=true)<br>

***

计算OOB样本到底有多少。<br>
假设bootstrap的数量N'=N，那么某个样本$(x_n,y_n)$是OOB的概率是：

$$(1-\frac1N)^N=\frac{1}{(\frac{N}{N-1})^N}=\frac{1}{(1+\frac{1}{N-1})^N}\approx \frac1e$$

其中，e是自然对数，N是原样本集的数量。<br>
由上述推导可得，每个$g_t$中，OOB数目大约是$\frac1eN$，即大约有三分之一的样本没有在bootstrap中被抽到。

然后，我们将OOB与之前介绍的Validation进行对比：<br>
![10](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/10_10.png?raw=true)<br>
在Validation表格中，蓝色的$D_{train}$用来得到不同的$g_m^-$，而红色的$D_{val}$用来验证各自的$g_m^-$。$D_{train}$与$D_{val}$没有交集，一般$D_{train}$是$D_{val}$的数倍关系。<br>
再看左边的OOB表格，之前我们也介绍过，蓝色的部分用来得到不同的$g_t$，而红色的部分是OOB样本。刚推导过，红色部分大约占N的$\frac1e$。<br>
如何使用OOB来验证G的好坏。方法是先看每一个样本$(x_n,y_n)$是哪些$g_t$的OOB资料，然后计算其在这些$g_t$上的表现，最后将所有样本的表现求平均即可。<br>
例如，样本$(x_N,y_N)$是$g_2$，$g_3$，$g_T$的OOB，则可以计算$(x_N,y_N)$在$G_N^-(x)$上的表现为：

$$G_N^-(x)=average(g_2,g_3,g_T)$$

像是Leave-One-Out Cross Validation，每次只对一个样本进行$g^-$的验证一样，只不过这里选择的是每个样本是哪些$g_t$的OOB，然后再分别进行$G_n^-(x)$的验证。每个样本都当成验证资料一次（与留一法相同），最后计算所有样本的平均表现：

$$E_{oob}(G)=\frac1N\sum_{n=1}^Nerr(y_n,G_n^-(x_n))$$

$E_{oob}(G)$估算的就是G的表现好坏。我们把$E_{oob}$称为bagging或者Random Forest的self-validation。

这种self-validation相比于validation来说还有一个优点就是它不需要重复训练。<br>
如下图左边所示，在通过$D_{val}$选择到表现最好的$g_{m^*}^-$之后，还需要在$D_{train}$和$D_{val}$组成的所有样本集D上重新对该模型$g_{m^*}^-$训练一次，以得到最终的模型系数。<br>
但是self-validation在调整随机森林算法相关系数并得到最小的$E_{oob}$之后，就完成了整个模型的建立，无需重新训练模型。<br>
随机森林算法中，self-validation在衡量G的表现上通常相当准确。<br>
![11](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/10_11.png?raw=true)<br>

---

### 3.Feature Selection
需要移除的特征分为两类：<br>
一类是冗余特征，即特征出现重复，例如“年龄”和“生日”；<br>
另一类是不相关特征，例如疾病预测的时候引入的“保险状况”。<br>
这种从d维特征到d'维特征的subset-transform $\Phi(x)$称为Feature Selection，最终使用这些d'维的特征进行模型训练。<br>
![12](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/10_12.png?raw=true)<br>

***

特征选择的优点 | 特征选择的缺点
---- | ---
提高效率，特征越少，模型越简单  | 筛选特征的计算量较大
正则化，防止特征过多出现过拟合  | 不同特征组合，也容易发生过拟合
去除无关特征，保留相关性大的特征，解释性强  | 容易选到无关特征，解释性差

![13](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/10_13.png?raw=true)<br>

***

在decision tree中，decision stump切割方式也是一种feature selection。<br>
可以通过计算出每个特征的重要性（即权重），然后再根据重要性的排序进行选择即可。<br>
![14](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/10_14.png?raw=true)<br>
这种方法在线性模型中比较容易计算。<br>
线性模型的score是由每个特征经过加权求和而得到的，而加权系数的绝对值$|w_i|$正好代表了对应特征$x_i$的重要性为多少。$|w_i|$越大，表示对应特征$x_i$越重要，则该特征应该被选择。<br>
w的值可以通过对已有的数据集$(x_i,y_i)$建立线性模型而得到。<br>
![15](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/10_15.png?raw=true)<br>

***

在非线性模型（Random Forest）下进行特征选择有点困难。

RF中，特征选择的核心思想是random test。<br>
random test的做法是对于某个特征，如果用另外一个随机值替代它之后的表现比之前更差，则表明该特征比较重要，所占的权重应该较大，不能用一个随机值替代。<br>
相反，如果随机值替代后的表现没有太大差别，则表明该特征不那么重要，可有可无。<br>
所以，通过**比较某特征被随机值替代前后的表现**，就能推断出该特征的权重和重要性。

random test中的随机值选择通常有两种方法：<br>
一是使用uniform或者gaussian抽取随机值替换原特征；<br>
一是通过permutation的方式将原来的所有N个样本的第i个特征值重新打乱分布（相当于重新洗牌）。<br>
比较而言，第二种方法更加科学，保证了特征替代值与原特征的分布是近似的（只是重新洗牌而已）。这种方法叫做permutation test（随机排序测试），即在计算第i个特征的重要性的时候，将N个样本的第i个特征重新洗牌，然后比较D和$D^{(p)}$表现的差异性。如果差异很大，则表明第i个特征是重要的。<br>
![16](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/10_16.png?raw=true)<br>
知道了permutation test的原理后，接下来要考虑的问题是如何衡量上图中的performance，即替换前后的表现。<br>
performance可以用$E_{oob}(G)$来衡量。<br>
但是，对于N个样本的第i个特征值重新洗牌重置的$D^{(p)}$，要对它进行重新训练，而且每个特征都要重复训练，然后再与原D的表现进行比较，过程非常繁琐。<br>
为了简化运算，RF的作者提出了一种方法，就是把permutation的操作从原来的training上移到了OOB validation上去，记为$E_{oob}(G^{(p)})\rightarrow E_{oob}^{(p)}(G)$。<br>
也就是说，在训练的时候仍然使用D，但是在OOB验证的时候，将所有的OOB样本的第i个特征重新洗牌，验证G的表现。<br>
这种做法大大简化了计算复杂度，在RF的feature selection中应用广泛。<br>
![17](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/10_17.png?raw=true)<br>

---

### 4.Random Forest in Action
一个二元分类的例子。<br>
如下图所示，左边是一个C&RT树没有使用bootstrap得到的模型分类效果，其中不同特征之间进行了随机组合，所以有斜线作为分类线；<br>
中间是由bootstrap（N'=N/2）后生成的一棵决策树组成的随机森林，图中加粗的点表示被bootstrap选中的点；<br>
右边是将一棵决策树进行bagging后的分类模型，效果与中间图是一样的，都是一棵树。<br>
![18](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/10_18.png?raw=true)<br>
<br>
当t=100，即选择了100棵树时，中间的模型是第100棵决策树构成的，还是只有一棵树；右边的模型是由100棵决策树bagging起来的，如下图所示：<br>
![19](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/10_19.png?raw=true)<br>
<br>
t=200时<br>
![20](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/10_20.png?raw=true)<br>
<br>
t=300时<br>
![21](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/10_21.png?raw=true)<br>
<br>
t=400时<br>
![22](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/10_22.png?raw=true)<br>
<br>
t=500<br>
![23](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/10_23.png?raw=true)<br>
<br>
t=600<br>
![24](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/10_24.png?raw=true)<br>
<br>
t=700<br>
![25](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/10_25.png?raw=true)<br>
<br>
t=800<br>
![26](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/10_26.png?raw=true)<br>
<br>
t=900<br>
![27](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/10_27.png?raw=true)<br>
<br>
t=1000<br>
![28](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/10_28.png?raw=true)<br>
<br>
随着树木个数的增加，我们发现，分界线越来越光滑而且得到了large-margin-like boundary，类似于SVM一样的效果。也就是说，树木越多，分类器的置信区间越大。

然后，我们再来看一个比较复杂的例子，二维平面上分布着许多离散点，分界线形如sin函数。当只有一棵树的时候（t=1），下图左边表示单一树组成的RF，右边表示所有树bagging组合起来构成的RF。因为只有一棵树，所以左右两边效果一致。<br>
![29](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/10_29.png?raw=true)<br>
<br>
t=6时<br>
![30](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/10_30.png?raw=true)<br>
<br>
t=11时<br>
![31](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/10_31.png?raw=true)<br>
<br>
t=16时<br>
![32](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/10_32.png?raw=true)<br>
<br>
t=21时<br>
![33](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/10_33.png?raw=true)<br>
可以看到，当RF由21棵树构成的时候，分界线就比较平滑了，而且它的边界比单一树构成的RF要robust得多，更加平滑和稳定。

最后，基于上面的例子，再让问题复杂一点：在平面上添加一些随机噪声。当t=1时，如下图所示：<br>
![34](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/10_34.png?raw=true)<br>
<br>
t=6<br>
![35](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/10_35.png?raw=true)<br>
<br>
t=11<br>
![36](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/10_36.png?raw=true)<br>
<br>
t=16<br>
![37](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/10_37.png?raw=true)<br>
<br>
t=21<br>
![38](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/10_38.png?raw=true)<br>
从上图中，我们发现21棵树的时候，随机noise的影响基本上能够修正和消除。这种bagging投票的机制能够保证较好的降噪性，从而得到比较稳定的结果。

经过以上三个例子，我们发现RF中，树的个数越多，模型越稳定越能表现得好。在实际应用中，应该尽可能选择更多的树。值得一提的是，RF的表现同时也与random seed有关，即随机的初始值也会影响RF的表现。<br>
![39](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/10_39.png?raw=true)<br>