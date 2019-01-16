# Kernel Support Vector Machine

---

### 1.Kernel Trick
推导的dual SVM是如下形式：<br>
![dualSVM](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/3_dualSVM.png?raw=true)<br>
其中$\alpha$是拉格朗日因子，共N个，这是我们要求解的，而条件共有N+1个。<br>
向量$Q_D$中的$q_{n,m}=y_ny_mz_n^Tz_m$，$z_n^Tz_m$的内积中会引入$\hat d$。即，如果$\hat d$很大，计算$z_n^Tz_m$的复杂度也会很高，会影响QP问题的计算效率。<br>
so,$q_{n,m}=y_ny_mz_n^Tz_m$这一步是计算的瓶颈所在。

其实问题的关键在于$z_n^Tz_m$内积求解上。我们知道，z是由x经过特征转换而来：

$$z_n^Tz_m=\Phi(x_n)\Phi(x_m)$$

如果从x空间来看的话，$z_n^Tz_m$分为两个步骤：<br>
1. 进行特征转换$\Phi(x_n)$和$\Phi(x_m)$；<br>
2. 计算$\Phi(x_n)$与$\Phi(x_m)$的内积。<br>
这种先转换再计算内积的方式，必然会引入$\hat d$参数，从而在$\hat d$很大的时候影响计算速度。<br>
尝试将两个步骤联合起来。<br>

***

我们先来看一个简单的例子，对于二阶多项式转换，各种排列组合为：<br>
![2nd](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/2_2nd.png?raw=true)<br>
<br>
内积推导：<br>
![neiji](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/2_neiji.png?raw=true)<br>
其中$x^Tx'$是x空间中特征向量的内积。<br>
所以，$\Phi_2(x)$与$\Phi_2(x')$的内积的复杂度由原来的$O(d^2)$变成$O(d)$，只与x空间的维度d有关，而与z空间的维度$\hat d$无关。<br>

我们把合并特征转换和计算内积这两个步骤的操作叫做Kernel Function，用大写字母K表示。例如刚刚讲的二阶多项式例子，它的kernel function为：<br>

$$K_{\Phi}(x,x')=\Phi(x)^T\Phi(x')$$

$$K_{\Phi_2}(x,x')=1+(x^Tx')+(x^Tx')^2$$

有了kernel function之后，我们来看看它在SVM里面如何使用。在dual SVM中，二次项系数$q_{n,m}$中有z的内积计算，就可以用kernel function替换：

$$q_{n,m}=y_ny_mz_n^Tz_m=y_ny_mK(x_n,x_m)$$

所以，直接计算出$K(x_n,x_m)$，再代入上式，就能得到$q_{n,m}$的值。

$q_{n,m}$值计算之后，就能通过QP得到拉格朗日因子$\alpha_n$。然后，下一步就是计算b（取$\alpha_n$>0的点，即SV），b的表达式中包含z，可以作如下推导：

$$b=y_s-w^Tz_s=y_s-(\sum_{n=1}^N\alpha_ny_nz_n)^Tz_s=y_s-\sum_{n=1}^N\alpha_ny_n(K(x_n,x_s))$$

这样得到的b就可以用kernel function表示，而与z空间无关。

最终我们要求的矩$g_{SVM}$可以作如下推导：

$$g_{SVM}(x)=sign(w^T\Phi(x)+b)=sign((\sum_{n=1}^N\alpha_ny_nz_n)^Tz+b)=sign(\sum_{n=1}^N\alpha_ny_n(K(x_n,x))+b)$$

至此，dual SVM中我们所有需要求解的参数都已经得到了，而且整个计算过程中都没有在z空间作内积，即与z无关。我们把这个过程称为kernel trick，也就是把特征转换和计算内积两个步骤结合起来，用kernel function来避免计算过程中受$\hat d$的影响，从而提高运算速度。

***
总结一下，引入kernel function后，svm算法变成：
![svm](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/3_sumsvm.png?raw=true)<br>

每个步骤的时间复杂度为:<br>
![times](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/3_times.png?raw=true)<br>

我们把这种引入kernel function的SVM称为kernel SVM，它是基于dual SVM推导而来的。kernel SVM同样只用SV（$\alpha_n$>0）就能得到最佳分类面，而且整个计算过程中摆脱了$\hat d$的影响，大大提高了计算速度。

---

### 2.Polynomial Kernel 
二次多项式的kernel形式多样。<br>
![poly](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/3_poly.png?raw=true)<br>

系数不同，内积就会有差异，就会代表不同的距离，最终可能会得到不同的SVM margin。
所以，系数不同，可能会得到不同的SVM分界线。<br>
![choose](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/3_choose.png?raw=true)<br>

不同的转换，对应到不同的几何距离<br>
![distances](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/3_distances.png?raw=true)<br>

引入$\zeta\geq 0$和$\gamma>0$，对于Q次多项式一般的kernel形式可表示为：<br>
![kernel](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/3_kernel.png?raw=true)<br>

所以，使用高阶的多项式kernel有两个优点:<br>
1.得到最大SVM margin，SV数量不会太多，分类面不会太复杂，防止过拟合，减少复杂度<br>
2.计算过程避免了对$\hat d$的依赖，大大简化了计算量。

---

### 3.Gaussian Kernel
如果是无限多维的转换$\Phi(x)$，也还能通过kernel的思想，来简化SVM的计算
先举个例子，简单起见，假设原空间是一维的，只有一个特征x，我们构造一个kernel function为高斯函数：

$$K(x,x')=e^{-(x-x')^2}$$

构造的过程正好与二次多项式kernel的相反，利用反推法，先将上式分解并做泰勒展开：
![taylor](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/3_taylor.png?raw=true)<br>
将构造的K(x,x')推导展开为两个$\Phi(x)$和$\Phi(x')$的乘积，其中：

$$\Phi(x)=e^{-x^2}\cdot (1,\sqrt \frac{2}{1!}x,\sqrt \frac{2^2}{2!}x^2,\cdots)$$
通过反推，我们得到了$\Phi(x)$，$\Phi(x)$是无限多维的，它就可以当成特征转换的函数，且$\hat d$是无限的。这种$\Phi(x)$得到的核函数即为Gaussian kernel。

更一般地，对于原空间不止一维的情况（d>1），引入缩放因子$\gamma>0$，它对应的Gaussian kernel表达式为：

$$K(x,x')=e^{-\gamma||x-x'||^2}$$

那么引入了高斯核函数，将有限维度的特征转换拓展到无限的特征转换中。根据本节课上一小节的内容，由K，计算得到$\alpha_n$和b，进而得到矩$g_{SVM}$。将其中的核函数K用高斯核函数代替，得到：

$$g_{SVM}(x)=sign(\sum_{SV}\alpha_ny_nK(x_n,x)+b)=sign(\sum_{SV}\alpha_ny_ne^{(-\gamma||x-x_n||^2)}+b)$$

通过上式可以看出，$g_{SVM}$有n个高斯函数线性组合而成，其中n是SV的个数。而且，每个高斯函数的中心都是对应的SV。通常我们也把高斯核函数称为径向基函数（Radial Basis Function, RBF）。

![rbf](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/3_rbf.png?raw=true)<br>

缩放因子$\gamma$取值不同，会得到不同的高斯核函数，hyperplanes不同，分类效果也有很大的差异。举个例子，$\gamma$分别取1, 10, 100时对应的分类效果如下：<br>
![hyper](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/3_hyper.png?raw=true)<br>
从图中可以看出，当$\gamma$比较小的时候，分类线比较光滑，当$\gamma$越来越大的时候，分类线变得越来越复杂和扭曲。<br>
因为$\gamma$越大，其对应的高斯核函数越尖瘦，那么有限个高斯核函数的线性组合就比较离散，分类效果并不好。<br>
所以，SVM也会出现过拟合现象，$\gamma$的正确选择尤为重要，不能太大。

---

### 4.Comparison of Kernels
对几种核进行比较。

Linear Kernel是最基本最简单的核，平面上对应一条直线，三维空间内对应一个平面。<br>
Linear Kernel可以使用Dual SVM中的QP直接计算得到。<br>
![linear](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/3_linear.png?raw=true)<br>

优点是计算简单、快速，可以直接使用QP快速得到参数值，而且从视觉上分类效果非常直观，便于理解<br>
缺点是如果数据不是线性可分的情况，Linear Kernel就不能使用了。<br>
![addis](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/3_addis.png?raw=true)<br>

***

Polynomial Kernel的hyperplanes是由多项式曲线构成。<br>
![hyperplanes](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/3_duo.png?raw=true)<br>
优点是阶数Q可以灵活设置，相比linear kernel限制更少，更贴近实际样本分布<br>
缺点是当Q很大时，K的数值范围波动很大，而且参数个数较多，难以选择合适的值。<br>
![hyperplanesad_disad](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/3_hyaddis.png?raw=true)<br>

***

对于Gaussian Kernel，表示为高斯函数形式。<br>
![gaussian](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/3_ga.png?raw=true)<br>
优点是边界更加复杂多样，能最准确地区分数据样本，数值计算K值波动较小，而且只有一个参数，容易选择<br>
缺点是由于特征转换到无限维度中，w没有求解出来，计算速度要低于linear kernel，而且可能会发生过拟合。<br>
![gaussianad_disad](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/3_gaaddis.png?raw=true)<br>

***

kernel代表的是两笔资料x和x'，特征变换后的相似性即内积。<br>
但是不能说任何计算相似性的函数都可以是kernel。<br>
有效的kernel还需满足几个条件：<br>
1.K是对称的<br>
2.K是半正定的<br>
<br>
这两个条件不仅是必要条件，同时也是充分条件。<br>
只要我们构造的K同时满足这两个条件，那它就是一个有效的kernel(Mercer定理)<br>
![kernel](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/3_kernel2.png?raw=true)<br>