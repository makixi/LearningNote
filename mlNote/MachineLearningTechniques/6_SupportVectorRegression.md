# Support Vector Regression

---

### 1.Kernel Ridge Regression 
对于任何包含正则项的L2-regularized linear model，它的最佳化解w都可以写成是z的线性组合形式，因此也就能引入核技巧，将模型kernelized化。<br>
![1](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/6_1.png?raw=true)<br>

***

Kernel Ridge Regression:<br>
![2](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/6_2.png?raw=true)<br>
<br>
最佳解$w_*$肯定是z的线性组合<br>
把$w_*=\sum_{n=1}^N\beta_nz_n$代入到ridge regression中，将z的内积用kernel替换，把求$w_*$的问题转化成求$\beta_n$的问题：<br>
![3](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/6_3.png?raw=true)<br>
第一项可以看成是$\beta_n$的正则项<br>
第二项可以看成是$\beta_n$的error function<br>
求解该式最小化对应的$\beta_n$值，解决了kernel ridge regression问题。

***

求解$\beta_n$的问题可以写成如下形式：<br>
![4](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/6_4.png?raw=true)<br>
$E_{aug}(\beta)$是关于$\beta$的二次多项式，要对$E_{aug}(\beta)$求最小化解，这种凸二次最优化问题，只需要先计算其梯度，再令梯度为零即可。<br>
令$\nabla E_{aug}(\beta)$等于零，得到：

$$\beta=(\lambda I+K)^{-1}y$$

且$(\lambda I+K)$的逆矩阵的逆矩阵一定存在。因为核函数K满足Mercer's condition，它是半正定的，且$\lambda>0$。<br>

***

比较linear ridge regression和kernel ridge regression的关系。<br>
![5](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/6_5.png?raw=true)<br>

左  | 右
---- | ---
linear ridge regression  | kernel ridge regression 
线性模型，只能拟合直线 | 非线性模型，更灵活
训练复杂度$O(d^3+d^2N)$ | 训练复杂度$O(N^3)$
预测复杂度$O(d)$ | 预测复杂度$O(N)$

![6](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/6_6.png?raw=true)<br>

---

### 2.Support Vector Regression Primal
kernel ridge regression应用在classification上叫做least-squares SVM(LSSVM)<br>

比较一下soft-margin Gaussian SVM和Gaussian LSSVM在分类上的差异：<br>
![7](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/6_7.png?raw=true)<br>
左边soft-margin Gaussian SVM的SV不多，而右边Gaussian LSSVM中基本上每个点都是SV。<br>
因为soft-margin Gaussian SVM中的$\alpha_n$大部分是等于零，$\alpha_n>0$的点只占少数，所以SV少。<br>
而对于LSSVM，$\beta$的解大部分都是非零值，所以对应的每个点基本上都是SV。<br>
SV太多会带来一个问题，就是做预测的矩$g(x)=\sum_{n=1}^N\beta_nK(x_n,x)$，如果$\beta_n$非零值较多，那么g的计算量也比较大，降低计算速度。<br>
so，soft-margin Gaussian SVM更有优势。<br>
![8](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/6_8.png?raw=true)<br>

***

可以通过一些方法得到sparse $\beta$，使得SV不会太多，从而得到和soft-margin SVM同样的分类效果。<br>
![9](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/6_9.png?raw=true)<br>
引入一个叫做Tube Regression的做法，即在分类线上下分别划定一个区域（中立区）<br>
如果数据点分布在这个区域内，则不算分类错误，只有误分在中立区域之外的地方才算error。<br>
假定中立区的宽度为$2\epsilon$，$\epsilon>0$,那么error measure就可以写成：$err(y,s)=max(0,|s-y|-\epsilon)$，对应上图中红色标注的距离。<br>

![10](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/6_10.png?raw=true)<br>

***

把tube regression中的error与squared error做个比较：<br>
![11](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/6_11.png?raw=true)<br>
将err(y,s)与s的关系曲线分别画出来：<br>
![12](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/6_12.png?raw=true)<br>
当|s-y|比较小即s比较接近y的时候，squared error与tube error是差不多大小的。<br>
而在|s-y|比较大的区域，squared error的增长幅度要比tube error大很多。<br>
error的增长幅度越大，表示越容易受到noise的影响，不利于最优化问题的求解。<br>
所以，从这个方面来看，tube regression的这种error function要更好一些。

***

L2-Regularized Tube Regression:<br>

![13](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/6_13.png?raw=true)<br>
上式，由于其中包含max项，并不是处处可微分的，所以不适合用GD/SGD来求解。<br>
而且，虽然满足representer theorem，有可能通过引入kernel来求解，但是也并不能保证得到sparsity $\beta$。<br>
从另一方面考虑，我们可以把这个问题转换为带条件的QP问题，仿照dual SVM的推导方法，引入kernel，得到KKT条件，从而保证解$\beta$是sparse的。<br>

![14](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/6_14.png?raw=true)<br>
所以，我们就可以把L2-Regularized Tube Regression写成跟SVM类似的形式：<br>
![15](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/6_15.png?raw=true)<br>

***

已经有了Standard Support Vector Regression的初始形式，这还是不是一个标准的QP问题。<br>
继续对该表达式做一些转化和推导：<br>
![16](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/6_16.png?raw=true)<br>

![17](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/6_17.png?raw=true)<br>
SVR的标准QP形式包含几个重要的参数：C和$\epsilon$。<br>
C表示的是regularization和tube violation之间的权衡。<br>
large C倾向于tube violation，small C则倾向于regularization。<br>
<br>
$\epsilon$表征了tube的区域宽度，即对错误点的容忍程度。<br>
$\epsilon$越大，则表示对错误的容忍度越大。<br>
$\epsilon$是可设置的常数，是SVR问题中独有的，SVM中没有这个参数。<br>
另外，SVR的QP形式共有$\hat{d}+1+2N$个参数，2N+2N个条件。<br>
![18](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/6_18.png?raw=true)<br>

---

### 3.Support Vector Regression Dual
先令拉格朗日因子$\alpha^{\bigvee}$和$\alpha^{\bigwedge}$，分别是与$\xi_n^{\bigvee}$和$\xi_n^{\bigwedge}$不等式相对应。<br>
![19](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/6_19.png?raw=true)<br>
然后，与SVM一样做同样的推导和化简，拉格朗日函数对相关参数偏微分为零，得到相应的KKT条件：<br>
![20](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/6_20.png?raw=true)<br>
通过观察SVM primal与SVM dual的参数对应关系，直接从SVR primal推导出SVR dual的形式<br>
![21](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/6_21.png?raw=true)<br>

***

SVR dual形式下推导的解w为：

$$w=\sum_{n=1}^N(\alpha_n^{\bigwedge}-\alpha_n^{\bigvee})z_n$$

相应的complementary slackness为：<br>
![22](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/6_22.png?raw=true)<br>

对于分布在tube中心区域内的点，满足$|w^Tz_n+b-y_n|<\epsilon$，此时忽略错误，$\xi_n^{\bigvee}$和$\xi_n^{\bigwedge}$都等于零。<br>
则complementary slackness两个等式的第二项均不为零，必然得到$\alpha_n^{\bigwedge}=0$和$\alpha_n^{\bigvee}=0$，即$\beta_n=\alpha_n^{\bigwedge}-\alpha_n^{\bigvee}=0$。

所以，对于分布在tube内的点，得到的解$\beta_n=0$，是sparse的。<br>
而分布在tube之外的点，$\beta_n\neq0$。<br>
至此，我们就得到了SVR的sparse解。

---

### 4.Summary of Kernel Models
![23](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/6_23.png?raw=true)<br>
上图中相应的模型也可以转化为dual形式，引入kernel，整体的框图如下：<br>
![24](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/6_24.png?raw=true)<br>