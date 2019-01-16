# Gradient Boosted Decision Tree 
使用Adaptive Boosting的方法来研究decision tree的一些算法和模型。

---

### 1.Adaptive Boosted Decision Tree 
Random Forest的算法：<br>
先通过bootstrapping“复制”原样本集D，得到新的样本集D'；<br>
然后对每个D'进行训练得到不同的decision tree和对应的$g_t$；<br>
最后再将所有的$g_t$通过uniform的形式组合起来，即以投票的方式得到G。<br>
这里采用的Bagging的方式，也就是把每个$g_t$的预测值直接相加。<br>
<br>
现在，如果将Bagging替换成AdaBoost，处理方式有些不同。<br>
首先每轮bootstrap得到的D'中每个样本会**赋予不同的权重**$u^{(t)}$；<br>
然后在每个decision tree中，利用这些权重训练得到最好的$g_t$；<br>
最后得出每个$g_t$所占的权重，线性组合得到G。<br>
这种模型称为AdaBoost-D Tree。<br>
![1](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/11_1.png?raw=true)<br>

***

但是在AdaBoost-DTree中需要注意的一点是每个样本的权重$u^{(t)}$。<br>
在Adaptive Boosting中进行了bootstrap操作，$u^{(t)}$表示D中每个样本在D'中出现的次数。但在决策树模型中，如C&RT算法中并没有引入$u^{(t)}$。<br>
那么，如何在决策树中引入这些权重$u^{(t)}$来得到不同的$g_t$而又不改变原来的决策树算法呢？

在Adaptive Boosting中，我们使用了weighted algorithm，形如：

$$E_{in}^u(h)=\frac1N\sum_{n=1}^Nu_n\cdot err(y_n,h(x_n))$$

每个犯错误的样本点乘以相应的权重，求和再平均，最终得到了$E_{in}^u(h)$。<br>
如果在决策树中使用这种方法，将当前分支下犯错误的点赋予权重，每层分支都这样做，会比较复杂，不易求解。<br>
为了简化运算，保持决策树算法本身的稳定性和封闭性，我们可以把决策树算法当成一个黑盒子，即不改变其结构，不对算法本身进行修改，而从数据来源D'上做一些处理。<br>
按照这种思想，我们来看权重u实际上表示该样本在bootstrap中出现的次数，反映了它出现的概率。<br>
那么可以根据u值，对原样本集D进行一次重新的随机sampling，也就是带权重的随机抽样。<br>
sampling之后，会得到一个新的D'，D'中每个样本出现的几率与它权重u所占的比例应该是差不多接近的。<br>
因此，**使用带权重的sampling操作**，得到了新的样本数据集D'，可以直接代入决策树进行训练，从而无需改变决策树算法结构。<br>
sampling可看成是bootstrap的反操作，这样就数据本身进行修改而不更改算法结构了。<br>
![2](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/11_2.png?raw=true)<br>

所以，AdaBoost-DTree结合了AdaBoost和DTree，但是做了一点小小的改变，就是**用sampling替代权重$u^{(t)}$**，效果是相同的。

![3](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/11_3.png?raw=true)<br>

使用sampling，将不同的样本集代入决策树中，得到不同的$g_t$。<br>
除此之外，我们还要确定每个$g_t$所占的权重$\alpha_t$。<br>
首先算出每个$g_t$的错误率$\epsilon_t$，然后计算权重：

$$\alpha_t=ln\ \diamond_t=ln \sqrt{\frac{1-\epsilon_t}{\epsilon_t}}$$

如果现在有一棵fully grown tree，由所有的样本$x_n$训练得到。<br>
若每个样本都不相同的话，一刀刀切割分支，直到所有的$x_n$都被完全分开。<br>
这时候，$E_{in}(g_t)=0$，加权的$E_{in}^u(g_t)=0$而且$\epsilon_t$也为0，从而得到权重$\alpha_t=\infty$。<br>
$\alpha_t=\infty$表示该$g_t$所占的权重无限大，相当于它一个就决定了G结构，是一种autocracy，而其它的$g_t$对G没有影响。<br>

![4](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/11_4.png?raw=true)<br>

显然$\alpha_t=\infty$不是我们想看到的，因为autocracy总是不好的，我们希望使用aggregation将不同的$g_t$结合起来，发挥集体智慧来得到最好的模型G。<br>
首先，我们来看一下什么原因造成了$\alpha_t=\infty$。<br>
有两个原因：一个是使用了所有的样本$x_n$进行训练；<br>
一个是树的分支过多，fully grown。<br>
<br>
针对这两个原因，我们可以对树做一些修剪（pruned），比如只使用一部分样本，这在sampling的操作中已经起到这类作用，因为必然有些样本没有被采样到。<br>
除此之外，我们还可以限制树的高度，让分支不要那么多，从而避免树fully grown。<br>
![5](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/11_5.png?raw=true)<br>
AdaBoost-DTree使用的是pruned DTree，也就是说将这些预测效果较弱的树结合起来，得到最好的G，避免出现autocracy。<br>
![6](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/11_6.png?raw=true)<br>

***

树只有1层高的时候，整棵树只有两个分支，切割一次即可。<br>
如果impurity是binary classification error的话，那么此时的AdaBoost-DTree就跟AdaBoost-Stump没什么两样。<br>
so,AdaBoost-Stump是AdaBoost-DTree的一种特殊情况。<br>
![7](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/11_7.png?raw=true)<br>
如果树高为1时，通常较难遇到$\epsilon_t=0$的情况，且一般不采用sampling的操作，而是直接将权重u代入到算法中。<br>
这是因为此时的AdaBoost-DTree就相当于是AdaBoost-Stump，而AdaBoost-Stump就是直接使用u来优化模型的。

---

### 2.Optimization View of AdaBoost
AdaBoost中的权重的迭代计算如下所示：<br>
![8](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/11_8.png?raw=true)<br>
之前对于incorrect样本和correct样本，$u_n^{(t+1)}$的表达式不同。现在，把两种情况结合起来，将$u_n^{(t+1)}$写成一种简化的形式：

$$u_n^{(t+1)}=u_n^{(t)}\cdot \diamond_t^{-y_ng_t(x_n)}=u_n^{(t)}\cdot exp(-y_n\alpha_tg_t(x_n))$$

其中，对于incorrect样本，$y_ng_t(x_n)<0$，对于correct样本，$y_ng_t(x_n)>0$。<br>
从上式可以看出，$u_n^{(t+1)}$由$u_n^{(t)}$与某个常数相乘得到。<br>
所以，最后一轮更新的$u_n^{(T+1)}$可以写成$u_n^{(1)}$的级联形式，我们之前令$u_n^{(1)}=\frac1N$，则有如下推导：

$$u_n^{(T+1)}=u_n^{(1)}\cdot \prod_{t=1}^Texp(-y_n\alpha_tg_t(x_n))=\frac1N\cdot exp(-y_n\sum_{t=1}^T\alpha_tg_t(x_n))$$

上式中$\sum_{t=1}^T\alpha_tg_t(x_n)$被称为voting score，最终的模型$G=sign(\sum_{t=1}^T\alpha_tg_t(x_n))$。<br>
可以看出，在AdaBoost中，$u_n^{(T+1)}$与$exp(-y_n(voting\ score\ on\ x_n))$成正比。<br>

![9](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/11_9.png?raw=true)<br>

***

voting score由许多$g_t(x_n)$乘以各自的系数$\alpha_t$线性组合而成。<br>
可以把$g_t(x_n)$看成是对$x_n$的特征转换$\phi_i(x_n)$，$\alpha_t$就是线性模型中的权重$w_i$。<br>
SVM中，w与$\phi (x_n)$的乘积再除以w的长度就是margin，即点到边界的距离。<br>
另外，乘积项再与$y_n$相乘，表示点的位置是在正确的那一侧还是错误的那一侧。<br>
所以，这里的voting score实际上可以看成是没有正规化（没有除以w的长度）的距离，即可以看成是该点到分类边界距离的一种衡量。<br>
从效果上说，距离越大越好，也就是说voting score要尽可能大一些。<br>

![10](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/11_10.png?raw=true)<br>

若voting score与$y_n$相乘，则表示一个有对错之分的距离。<br>
也就是说，如果二者相乘是负数，则表示该点在错误的一边，分类错误；如果二者相乘是正数，则表示该点在正确的一边，分类正确。<br>
所以，我们算法的目的就是让$y_n$与voting score的乘积是正的，而且越大越好。<br>
那么在刚刚推导的$u_n^{(T+1)}$中，得到$exp(-y_n(voting\ score))$越小越好，从而得到$u_n^{(T+1)}$越小越好。<br>
也就是说，如果voting score表现不错，与$y_n$的乘积越大的话，那么相应的$u_n^{(T+1)}$应该是最小的。<br>

![11](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/11_11.png?raw=true)<br>

那么在AdaBoost中，随着每轮学习的进行，每个样本的$u_n^{(t)}$是逐渐减小的，直到$u_n^{(T+1)}$最小。以上是从单个样本点来看的。<br>
总体来看，所有样本的$u_n^{(T+1)}$之和应该也是最小的。<br>
我们的目标就是在最后一轮（T+1）学习后，让所有样本的$u_n^{(T+1)}$之和尽可能地小。<br>
$u_n^{(T+1)}$之和表示为如下形式：<br>

![12](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/11_12.png?raw=true)<br>

上式中，$\sum_{t=1}^T\alpha_tg_t(x_n)$被称为linear score，用s表示。<br>
对于0/1 error：若ys<0，则$err_{0/1}=1$；若ys>=0，则$err_{0/1}=0$。<br>
对于指数error，即$\hat{err}{ADA}(s,y)=exp(-ys)$，随着ys的增加，error单调下降，且始终落在0/1 error折线的上面。<br>
$\hat{err}{ADA}(s,y)$可以看成是0/1 error的上界。所以，我们可以使用$\hat{err}{ADA}(s,y)$来替代0/1 error，能达到同样的效果。<br>
从这点来说，$\sum{n=1}^Nu_n^{(T+1)}$可以看成是一种error measure，而我们的目标就是让其最小化，求出最小值时对应的各个$\alpha_t$和$g_t(x_n)$。<br>

![13](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/11_13.png?raw=true)<br>

如何让$\sum_{n=1}^Nu_n^{(T+1)}$取得最小值，思考是否能用梯度下降（gradient descent）的方法来进行求解。<br>
gradient descent的核心是在某点处做一阶泰勒展开：<br>
![14](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/11_14.png?raw=true)<br>

其中，$w_t$是泰勒展开的位置，v是所要求的下降的最好方向，它是梯度$\nabla E_{in}(w_t)$的反方向，而$\eta$是每次前进的步长。则每次沿着当前梯度的反方向走一小步，就会不断逼近谷底（最小值）。这就是梯度下降算法所做的事情。

现在，对$\check{E}_{ADA}$做梯度下降算法处理，区别是这里的方向是一个函数$g_t$，而不是一个向量$w_t$。<br>
其实，函数和向量的唯一区别就是一个下标是连续的，另一个下标是离散的，二者在梯度下降算法应用上并没有大的区别。<br>
因此，按照梯度下降算法的展开式，做出如下推导：<br>
![15](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/11_15.png?raw=true)<br>

上式中，$h(x_n)$表示当前的方向，它是一个矩，$\eta$是沿着当前方向前进的步长。<br>
我们要求出这样的$h(x_n)$和$\eta$，使得$\check{E}_{ADA}$是在不断减小的。<br>
当$\check{E}_{ADA}$取得最小值的时候，那么所有的方向即最佳的$h(x_n)$和$\eta$就都解出来了。<br>
上述推导使用了在$-y_n\eta h(x_n)=0$处的一阶泰勒展开近似。<br>
这样经过推导之后，$\check{E}_{ADA}$被分解为两个部分，一个是前N个u之和$\sum{n=1}^Nu_n^{(t)}$，也就是当前所有的$E_{in}$之和；另外一个是包含下一步前进的方向$h(x_n)$和步进长度$\eta$的项$-\eta\sum_{n=1}^Nu_n^{(t)}y_nh(x_n)$。$\check{E}_{ADA}$的这种形式与gradient descent的形式基本是一致的。

那么接下来，如果要最小化$\check{E}_{ADA}$的话，就要让第二项$-\eta\sum_{n=1}^Nu_n^{(t)}y_nh(x_n)$越小越好。<br>
则我们的目标就是找到一个好的$h(x_n)$（即好的方向）来最小化$\sum_{n=1}^Nu_n^{(t)}(-y_nh(x_n))$，此时先忽略步进长度$\eta$。<br>
![16](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/11_16.png?raw=true)<br>

对于binary classification，$y_n$和$h(x_n)$均限定取值-1或+1两种。我们对$\sum_{n=1}^Nu_n^{(t)}(-y_nh(x_n))$做一些推导和平移运算：<br>

![17](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/11_17.png?raw=true)<br>

最终$\sum_{n=1}^Nu_n^{(t)}(-y_nh(x_n))$化简为两项组成，一项是$-\sum_{n=1}^Nu_n^{(t)}$；另一项是$2E_{in}^{u(t)}(h)\cdot N$。<br>
则最小化$\sum_{n=1}^Nu_n^{(t)}(-y_nh(x_n))$就转化为最小化$E_{in}^{u(t)}(h)$。<br>
要让$E_{in}^{u(t)}(h)$最小化，正是由AdaBoost中的base algorithm所做的事情。<br>
AdaBoost中的base algorithm正好帮我们找到了梯度下降中下一步最好的函数方向。<br>

![18](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/11_18.png?raw=true)<br>

以上就是从数学上，从gradient descent角度验证了AdaBoost中使用base algorithm得到的$g_t$就是让$\check{E}_{ADA}$减小的方向，只不过这个方向是一个函数而不是向量。

在解决了方向问题后，需要考虑步进长度$\eta$如何选取。<br>
方法是在确定方向$g_t$后，选取合适的$\eta$，使$\check{E}_{ADA}$取得最小值。<br>
把$\check{E}_{ADA}$看成是步进长度$\eta$的函数，目标是找到$\check{E}_{ADA}$最小化时对应的$\eta$值。<br>

![19](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/11_19.png?raw=true)<br>

目的是找到在最佳方向上的最大步进长度，也就是steepest decent。<br>
先把$\check{E}_{ADA}$表达式写下来：

$$\check{E}_{ADA}=\sum_{n=1}^Nu_n^{(t)}exp(-y_n\eta g_t(x_n))$$

上式中，有两种情况需要考虑：

$y_n=g_t(x_n)$：$u_n^{(t)}exp(-\eta)$ correct

$y_n\neq g_t(x_n)$：$u_n^{(t)}exp(+\eta)$ incorrect

经过推导，可得：

$$\check{E}{ADA}=(\sum{n=1}^Nu_n^{(t)})\cdot ((1-\epsilon_t)exp(-\eta)+\epsilon_t\ exp(+\eta))$$

![20](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/11_20.png?raw=true)<br>

然后对$\eta$求导，令$\frac{\partial \check{E}_{ADA}}{\partial \eta}=0$，得：

$$\eta_t=ln\sqrt{\frac{1-\epsilon_t}{\epsilon_t}}=\alpha_t$$

由此看出，最大的步进长度就是$\alpha_t$，即AdaBoost中计算$g_t$所占的权重。<br>
所以，AdaBoost算法所做的其实是在gradient descent上找到下降最快的方向和最大的步进长度。<br>
这里的方向就是$g_t$，它是一个函数，而步进长度就是$\alpha_t$。<br>
也就是说，在AdaBoost中确定$g_t$和$\alpha_t$的过程就相当于在gradient descent上寻找最快的下降方向和最大的步进长度。

---

### 3.Gradient Boosting
从gradient descent的角度来重新介绍了AdaBoost的最优化求解方法。整个过程可以概括为：<br>
![21](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/11_21.png?raw=true)<br>

以上是针对binary classification问题。<br>
如果往更一般的情况进行推广，这种情况下的GradientBoost可以写成如下形式：

![22](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/11_22.png?raw=true)<br>

仍然按照gradient descent的思想，上式中，$h(x_n)$是下一步前进的方向，$\eta$是步进长度。<br>
此时的error function不是前面所讲的exp了，而是任意的一种error function。<br>
因此，对应的hypothesis也不再是binary classification，最常用的是实数输出的hypothesis，例如regression。<br>
最终的目标也是求解最佳的前进方向$h(x_n)$和最快的步进长度$\eta$。

![23](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/11_23.png?raw=true)<br>

regression的GradientBoost问题:<br>

![24](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/11_24.png?raw=true)<br>

利用梯度下降的思想，进行一阶泰勒展开，写成梯度的形式：<br>

![25](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/11_25.png?raw=true)<br>

上式中，由于regression的error function是squared的，所以，对s的导数就是$2(s_n-y_n)$。其中标注灰色的部分表示常数，对最小化求解并没有影响，所以可以忽略。<br>
要使上式最小化，只要令$h(x_n)$是梯度$2(s_n-y_n)$的反方向就行了，即$h(x_n)=-2(s_n-y_n)$。<br>
但是直接这样赋值，并没有对$h(x_n)$的大小进行限制，一般不直接利用这个关系求出$h(x_n)$。 

![26](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/11_26.png?raw=true)<br>

实际上$h(x_n)$的大小并不重要，因为有步进长度$\eta$。<br>
那么，我们上面的最小化问题中需要对$h(x_n)$的大小做些限制。<br>
限制$h(x_n)$的一种简单做法是把$h(x_n)$的大小当成一个惩罚项（$h^2(x_n)$）添加到上面的最小化问题中，这种做法与regularization类似。<br>
经过推导和整理，忽略常数项，我们得到最关心的式子是：

$$min\ \sum_{n=1}^N((h(x_n)-(y_n-s_n))^2)$$

上式是一个完全平方项之和，$y_n-s_n$表示当前第n个样本真实值和预测值的差，称之为余数。<br>
余数表示当前预测能够做到的效果与真实值的差值是多少。<br>
那么，如果我们想要让上式最小化，求出对应的$h(x_n)$的话，只要让$h(x_n)$尽可能地接近余数$y_n-s_n$即可。<br>
在平方误差上尽可能接近其实很简单，就是使用regression的方法，对所有N个点$(x_n,y_n-s_n)$做squared-error的regression，得到的回归方程就是我们要求的$g_t(x_n)$。

![27](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/11_27.png?raw=true)<br>

以上就是使用GradientBoost的思想来解决regression问题的方法，其中应用了一个非常重要的概念，就是余数$y_n-s_n$。<br>
根据这些余数做regression，得到好的矩$g_t(x_n)$，方向函数$g_t(x_n)$也就是由余数决定的。

![28](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/11_28.png?raw=true)<br>

在求出最好的方向函数$g_t(x_n)$之后，就要来求相应的步进长度$\eta$。表达式如下：

![29](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/11_29.png?raw=true)<br>

同样，对上式进行推导和化简，得到如下表达式：

![30](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/11_30.png?raw=true)<br>

上式中也包含了余数$y_n-s_n$，其中$g_t(x_n)$可以看成是$x_n$的特征转换，是已知量。<br>
那么，如果想要让上式最小化，求出对应的$\eta$的话，只要让$\eta g_t(x_n)$尽可能地接近余数$y_n-s_n$即可。<br>
这也是一个regression问题，而且是一个很简单的形如y=ax的线性回归，只有一个未知数$\eta$。只要对所有N个点$(\eta g_t(x_n),y_n-s_n)$做squared-error的linear regression，利用梯度下降算法就能得到最佳的$\eta$。

将上述这些概念合并到一起，就得到了一个最终的演算法Gradient Boosted Decision Tree(GBDT)。<br>
在计算方向函数$g_t$的时候，是对所有N个点$(x_n,y_n-s_n)$做squared-error的regression。那么这个回归算法就可以是决策树C&RT模型（决策树也可以用来做regression）。这样，就引入了Decision Tree，并将GradientBoost和Decision Tree结合起来，构成了真正的GBDT算法。<br>
<br>
GBDT算法的基本流程图如下所示：

![31](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/11_31.png?raw=true)<br>

$s_n$的初始值一般均设为0，即$s_1=s_2=\cdots =s_N=0$。<br>
每轮迭代中，方向函数$g_t$通过C&RT算法做regression，进行求解；步进长度$\eta$通过简单的单参数线性回归进行求解；然后每轮更新$s_n$的值，即$s_n\leftarrow s_n+\alpha_tg_t(x_n)$。<br>
T轮迭代结束后，最终得到$G(x)=\sum_{t=1}^T\alpha_tg_t(x)$。

可以说GBDT就是AdaBoost-DTree的regression版本。

![32](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/11_32.png?raw=true)<br>

---

### 4.Summary ofAggregation Models
blending就是将所有已知的$g_t$ aggregate结合起来，发挥集体的智慧得到G。值得注意的一点是这里的$g_t$都是已知的。<br>
<br>
blending通常有三种形式：<br>
1.uniform：简单地计算所有$g_t$的平均值<br>
2.non-uniform：所有$g_t$的线性组合<br>
3.conditional：所有$g_t$的非线性组合<br>

其中，uniform采用投票、求平均的形式更注重稳定性；<br>
而non-uniform和conditional追求的更复杂准确的模型，但存在过拟合的危险。

![33](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/11_33.png?raw=true)<br>

blending是建立在所有$g_t$已知的情况。<br>
那如果所有$g_t$未知的情况，对应的就是learning模型，做法就是一边学$g_t$，一边将它们结合起来。<br>
<br>
learning通常也有三种形式（与blending的三种形式一一对应）：<br>
1.Bagging：通过bootstrap方法，得到不同$g_t$，计算所有$g_t$的平均值<br>
2.AdaBoost：通过bootstrap方法，得到不同$g_t$，所有$g_t$的线性组合<br>
3.Decision Tree：通过数据分割的形式得到不同的$g_t$，所有$g_t$的非线性组合<br>

将AdaBoost延伸到另一个模型GradientBoost。<br>
对于regression问题，GradientBoost通过residual fitting的方式得到最佳的方向函数$g_t$和步进长度$\eta$。

![34](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/11_34.png?raw=true)<br>

除了这些基本的aggregation模型之外，我们还可以把某些模型结合起来得到新的aggregation模型。<br>
例如，Bagging与Decision Tree结合起来组成了Random Forest。<br>
Random Forest中的Decision Tree是比较“茂盛”的树，即每个树的$g_t$都比较强一些。<br>
AdaBoost与Decision Tree结合组成了AdaBoost-DTree。<br>
AdaBoost-DTree的Decision Tree是比较“矮弱”的树，即每个树的$g_t$都比较弱一些，由AdaBoost将所有弱弱的树结合起来，让综合能力更强。<br>
同样，GradientBoost与Decision Tree结合就构成了经典的算法GBDT。

![35](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/11_35.png?raw=true)<br>

Aggregation的核心是将所有的$g_t$结合起来，融合到一起，即集体智慧的思想。这种做法之所以能得到很好的模型G，是因为aggregation具有两个方面的优点：cure underfitting和cure overfitting。

第一，aggregation models有助于防止欠拟合（underfitting）。它把所有比较弱的$g_t$结合起来，利用集体智慧来获得比较好的模型G。aggregation就相当于是feature transform，来获得复杂的学习模型。

第二，aggregation models有助于防止过拟合（overfitting）。它把所有$g_t$进行组合，容易得到一个比较中庸的模型，类似于SVM的large margin一样的效果，从而避免一些极端情况包括过拟合的发生。从这个角度来说，aggregation起到了regularization的效果。

由于aggregation具有这两个方面的优点，所以在实际应用中aggregation models都有很好的表现。

![36](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/11_36.png?raw=true)<br>

---

### 5.Summary 
Gradient Boosted Decision Tree。<br>
首先讲如何将AdaBoost与Decision Tree结合起来，即通过sampling和pruning的方法得到AdaBoost-D Tree模型。<br>
然后，我们从optimization的角度来看AdaBoost，找到好的hypothesis也就是找到一个好的方向，找到权重$\alpha$也就是找到合适的步进长度。<br>
接着，我们从binary classification的0/1 error推广到其它的error function，从Gradient Boosting角度推导了regression的squared error形式。Gradient Boosting其实就是不断迭代，做residual fitting。并将其与Decision Tree算法结合，得到了经典的GBDT算法。<br>
最后，我们将所有的aggregation models做了总结和概括，这些模型有的能防止欠拟合有的能防止过拟合，应用十分广泛。