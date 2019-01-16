# Soft Margin Support Vector Machine
之前讲的这些方法都是Hard-Margin SVM，即必须将所有的样本都分类正确才行。<br>
这往往需要更多更复杂的特征转换，甚至造成过拟合。<br>

***

这次的Soft-Margin SVM，目的是让分类错误的点越少越好，而不是必须将所有点分类正确，也就是允许有noise存在。<br>
这种做法很大程度上不会使模型过于复杂，不会造成过拟合，而且分类效果是令人满意的。

---

### 1.Motivation and Primal Problem
SVM同样可能会造成overfit。<br>
原因有两个<br>
一个是由于我们的SVM模型（即kernel）过于复杂，转换的维度太多，过于powerful了<br>
另外一个是由于我们坚持要将所有的样本都分类正确，即不允许错误存在，造成模型过于复杂。<br>

***

可以借鉴pocket（pocket的思想不是将所有点完全分开，而是找到一条分类线能让分类错误的点最少）
![pocket](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/4_pocket.png?raw=true)<br>

***

为了引入允许犯错误的点，我们将Hard-Margin SVM的目标和条件做一些结合和修正，转换为如下形式：<br>
![goal](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/4_newgoal.png?raw=true)<br>
对于分类正确的点，仍需满足$y_n(w^Tz_n+b)\geq 1$<br>
对于noise点，满足$y_n(w^Tz_n+b)\geq -\infty$<br>
修正后的目标除了$\frac12w^Tw$项，还添加了$y_n\neq sign(w^Tz_n+b)$，即noise点的个数<br>
参数C的引入是为了权衡目标第一项和第二项的关系，即权衡large margin和noise tolerance的关系。

***

两个条件合并，得到：<br>
![combine](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/4_combine.png?raw=true)<br>
这个式子存在两个不足的地方。<br>
首先，最小化目标中第二项是非线性的，不满足QP的条件，所以无法使用dual或者kernel SVM来计算。<br>
其次，对于犯错误的点，有的离边界很近，即error小，而有的离边界很远，error很大，上式的条件和目标没有区分small error和large error。这种分类效果是不完美的。<br>

***

为了改进不足，作如下修正：<br>
![re](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/4_re.png?raw=true)<br>
修正后的表达式中，我们引入了新的参数$\xi_n$来表示每个点犯错误的程度值，$\xi_n\geq0$。<br>
通过使用error值的大小代替是否有error，让问题变得易于求解，满足QP形式要求。<br>
这种方法类似于我们在机器学习基石笔记中介绍的0/1 error和squared error。<br>
这种soft-margin SVM引入新的参数$\xi$。

***

现在，最终的Soft-Margin SVM的目标为：

$$min(b,w,\xi)\ \frac12w^Tw+C\cdot\sum_{n=1}^N\xi_n$$

条件是：

$$y_n(w^Tz_n+b)\geq 1-\xi_n$$

$$\xi_n\geq0$$

其中，$\xi_n$表示每个点犯错误的程度，$\xi_n=0$，表示没有错误，$\xi_n$越大，表示错误越大，即点距离边界（负的）越大。<br>
参数C表示尽可能选择宽边界和尽可能不要犯错两者之间的权衡，因为边界宽了，往往犯错误的点会增加。<br>
large C表示希望得到更少的分类错误，即不惜选择窄边界也要尽可能把更多点正确分类<br>
small C表示希望得到更宽的边界，即不惜增加错误点个数也要选择更宽的分类边界。<br>

***

与之对应的QP问题中，由于新的参数$\xi_n$的引入，总共参数个数为$\hat d+1+N$，限制条件添加了$\xi_n\geq0$，则总条件个数为2N。
![qp](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/4_qp.png?raw=true)<br>

---

### 2.Dual Problem
Soft-Margin SVM的原始形式：<br>
![smsvm](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/4_smsvm.png?raw=true)<br>

***

构造一个拉格朗日函数。<br>
![lagrange](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/4_lagrange.png?raw=true)<br>

***

利用Lagrange dual problem，将Soft-Margin SVM问题转换为如下形式：<br>
![newsvm](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/4_newsvm.png?raw=true)<br>

***

根据之前介绍的KKT条件，我们对上式进行简化。上式括号里面的是对拉格朗日函数$L(b,w,\xi,\alpha,\beta)$计算最小值。那么根据梯度下降算法思想：最小值位置满足梯度为零。

我们先对$\xi_n$做偏微分：

$$\frac{\partial L}{\partial \xi_n}=0=C-\alpha_n-\beta_n$$

根据上式，得到$\beta_n=C-\alpha_n$，因为有$\beta_n\geq0$，所以限制$0\leq\alpha_n\leq C$<br>
将$\beta_n=C-\alpha_n$代入到dual形式中并化简，我们发现$\beta_n$和$\xi_n$都被消去了：<br>
![dairu](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/4_dairu.png?raw=true)<br>

***

分别令拉格朗日函数L对b和w的偏导数为零，分别得到：

$$\sum_{n=1}^N\alpha_ny_n=0$$

$$w=\sum_{n=1}^N\alpha_ny_nz_n$$

经过化简和推导，最终标准的Soft-Margin SVM的Dual形式如下图所示：<br>
![dualsvm](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/4_dualsvm.png?raw=true)<br>
Soft-Margin SVM Dual与Hard-Margin SVM Dual基本一致，只有一些条件不同。<br>
Hard-Margin SVM Dual中$\alpha_n\geq0$，而Soft-Margin SVM Dual中$0\leq\alpha_n\leq C$，且新的拉格朗日因子$\beta_n=C-\alpha_n$。<br>
在QP问题中，Soft-Margin SVM Dual的参数$\alpha_n$同样是N个，但是，条件由Hard-Margin SVM Dual中的N+1个变成2N+1个，这是因为多了N个$\alpha_n$的上界条件。

---

### 3.Messages behind Soft-Margin SVM
Soft-Margin SVM Dual计算$\alpha_n$的方法过程：<br>
![jisuan](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/4_jisuan.png?raw=true)<br>
在Hard-Margin SVM Dual中，有complementary slackness条件：$\alpha_n(1-y_n(w^Tz_n+b))=0$，找到SV，即$\alpha_s>0$的点，计算得到$b=y_s-w^Tz_s$。<br>

![svm](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/4_svm.png?raw=true)<br>

***

对于Soft-Margin Gaussian SVM，C分别取1，10，100时，相应的margin如下图所示：<br>
![c](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/4_c.png?raw=true)<br>
C越小，越倾向于得到粗的margin，增加分类错误的点；C越大，越倾向于得到高的分类正确率。<br>
我们发现，当C值很大的时候，虽然分类正确率提高，但很可能把noise也进行了处理，从而可能造成过拟合。<br>
也就是说Soft-Margin Gaussian SVM同样可能会出现过拟合现象，所以参数$(\gamma,C)$的选择非常重要。

***

在Soft-Margin SVM Dual中，根据$\alpha_n$的取值，就可以推断数据点在空间的分布情况:<br>
![fenbu](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/4_fenbu.png?raw=true)<br>
$$\alpha_n(1-\xi_n-y_n(w^Tz_n+b))=0$$

$$\beta_n\xi_n=(C-\alpha_n)\xi=0$$

若$\alpha_n=0$，得$\xi_n=0$。$\xi_n=0$表示该点没有犯错，$\alpha_n=0$表示该点不是SV。所以对应的点在margin之外（或者在margin上），且均分类正确。

若$0<\alpha_n<C$，得$\xi_n=0$，且$y_n(w^Tz_n+b)=1$。$\xi_n=0$表示该点没有犯错，$y_n(w^Tz_n+b)=1$表示该点在margin上。这些点即free SV，确定了b的值。

若$\alpha_n=C$，不能确定$\xi_n$是否为零，且得到$1-y_n(w^Tz_n+b)=\xi_n$，这个式表示该点偏离margin的程度，$\xi_n$越大，偏离margin的程度越大。只有当$\xi_n=0$时，该点落在margin上。所以这种情况对应的点在margin之内负方向（或者在margin上），有分类正确也有分类错误的。这些点称为bounded SV。

---

### 4.Model Selection
对于Gaussian SVM，不同的参数$(C,\gamma)$，会得到不同的margin：<br>
![gaussian margin](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/4_gamargin.png?raw=true)<br>
其中横坐标是C逐渐增大的情况，纵坐标是$\gamma$逐渐增大的情况。不同的$(C,\gamma)$组合，margin的差别很大。用validation选择最好的$(C,\gamma)$等参数。

***

由不同$(C,\gamma)$等参数得到的模型在验证集上进行cross validation，选取$E_{cv}$最小的对应的模型就可以了<br>
例如上图中各种$(C,\gamma)$组合得到的$E_{cv}$如下图所示：<br>
![ecv](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/4_ecv.png?raw=true)<br>

V-Fold cross validation的一种极限就是Leave-One-Out CV，也就是验证集只有一个样本。对于SVM问题，它的验证集Error满足：

$$E_{loocv}\leq \frac{SV}{N}$$

那么，对于non-SV的点，它的$g^-=g$，即对第N个点，它的Error必然为零：

$$e_{non-SV}=err(g^-,non-SV)=err(g,non-SV)=0$$

另一方面，假设第N个点$\alpha_N\neq0$，即对于SV的点，它的Error可能是0，也可能是1，必然有：

$$e_{SV}\leq1$$

综上所述，即证明了$E_{loocv}\leq \frac{SV}{N}$。这符合我们之前得到的结论，即只有SV影响margin，non-SV对margin没有任何影响，可以舍弃。

一般来说，SV越多，表示模型可能越复杂，越有可能会造成过拟合。所以，通常选择SV数量较少的模型，然后在剩下的模型中使用cross-validation，比较选择最佳模型。