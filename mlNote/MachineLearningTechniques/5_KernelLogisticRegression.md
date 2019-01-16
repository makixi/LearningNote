# Kernel Logistic Regression

---

### 1.Soft-Margin SVM as Regularized Model
最早有Hard-Margin Primal,然后推导出Hard-Margin Dual形式。<br>
后来，为了允许有错误点存在（或者noise），也为了避免模型太过复杂化，造成过拟合，建立了Soft-Margin Primal的数学表达式，并引入了新的参数C作为权衡因子，然后也推导了其Soft-Margin Dual形式。<br>
![soft](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/5_softad.png?raw=true)<br>

***

![zh](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/5_zh.png?raw=true)<br>
$\xi_n$描述的是点$(x_n,y_n)$ 距离$y_n(w^Tz_n+b)=1$的边界有多远。<br>
第一种情况是violating margin，即不满足$y_n(w^Tz_n+b)\geq1$。那么$\xi_n$可表示为：$\xi_n=1-y_n(w^Tz_n+b)>0$。<br>
第二种情况是not violating margin，即点$(x_n,y_n)$ 在边界之外，满足$y_n(w^Tz_n+b)\geq1$的条件，此时$\xi_n=0$。<br>
<br>
我们可以将两种情况整合到一个表达式中，对任意点：

$$\xi_n=max(1-y_n(w^Tz_n+b),0)$$

上式表明，如果有voilating margin，则$1-y_n(w^Tz_n+b)>0$，$\xi_n=1-y_n(w^Tz_n+b)$<br>
如果not violating margin，则$1-y_n(w^Tz_n+b)<0$，$\xi_n=0$。<br>
<br>
整合之后，我们可以把Soft-Margin SVM的最小化问题写成如下形式：

$$\frac12w^Tw+C\sum_{n=1}^Nmax(1-y_n(w^Tz_n+b),0)$$

经过这种转换之后，表征犯错误值大小的变量$\xi_n$就被消去了，转而由一个max操作代替。

***

![goal](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/5_goal.png?raw=true)<br>
L2 Regularization中的$\lambda$和Soft-Margin SVM中的C也是相互对应的，$\lambda$越大，w会越小，Regularization的程度就越大；C越小，$\hat{E_{in}}$会越大，相应的margin就越大。<br>
所以说增大C，或者减小$\lambda$，效果是一致的，Large-Margin等同于Regularization，都起到了防止过拟合的作用。

---

### 2.SVM versus Logistic Regression
我们已经把Soft-Margin SVM转换成无条件的形式：<br>
![wu](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/5_wu.png?raw=true)<br>
$max(1-y_n(w^Tz_n+b),0)$倍设置为$\hat{err}$<br>

***

![01](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/5_01.png?raw=true)<br>
对于$err_{0/1}$，它的linear score $s=w^Tz_n+b$<br>
当$ys\geq0$时，$err_{0/1}=0$<br>
当$ys<0$时，$err_{0/1}=1$，呈阶梯状。<br>
<br>
对于$\hat{err}$，当$ys\geq0$时，$err_{0/1}=0$<br>
当$ys<0$时，$err_{0/1}=1-ys$，呈折线状。<br>
<br>
$\hat{err}_{svm}$始终在$err_{0/1}$的上面，则$\hat{err}_{svm}$可作为$err{0/1}$的上界。<br>
所以，可以使用$\hat{err}_{svm}$来代替$err{0/1}$，解决二元线性分类问题，而且$\hat{err}_{svm}$是一个凸函数，使它在最佳化问题中有更好的性质。

***

![sce](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/5_sce.png?raw=true)<br>
逻辑回归中，$err_{sce}=log_2(1+exp(-ys))$，当ys=0时，$err_{sce}=1$。

$err_{sce}$也是$err_{0/1}$的上界，而$err_{sce}$与$\hat{err}_{svm}$也是比较相近的。<br>
因为当ys趋向正无穷大的时候，$err_{sce}$和$\hat{err}_{svm}$都趋向于零；<br>
当ys趋向负无穷大的时候，$err_{sce}$和$\hat{err}_{svm}$都趋向于正无穷大。<br>
可以把SVM看成是L2-regularized logistic regression。

***

![sum](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/5_sum.png?raw=true)<br>
**PLA**是相对简单的一个模型，对应的是$err_{0/1}$<br>
通过不断修正错误的点来获得最佳分类线<br>
优点是简单快速<br>
缺点是只对线性可分的情况有用，线性不可分的情况需要用到pocket算法。<br>
<br>
Logistic Regression对应的是$err_{sce}$，通常使用GD/SGD算法求解最佳分类线。<br>
优点是凸函数$err_{sce}$便于最优化求解，而且有regularization作为避免过拟合的保证<br>
缺点是$err_{sce}$作为$err_{0/1}$的上界，当ys很小（负值）时，上界变得更宽松，不利于最优化求解。<br>
<br>
Soft-Margin SVM对应的是$\hat{err}{svm}$，通常使用QP求解最佳分类线。<br>
优点和Logistic Regression一样，凸优化问题计算简单而且分类线比较“粗壮”一些<br>
缺点也和Logistic Regression一样，当ys很小（负值）时，上界变得过于宽松。<br>
<br>
Logistic Regression和Soft-Margin SVM都是在最佳化$err{0/1}$的上界而已。

---

### 3.SVM for Soft Binary Classification
第一种简单的方法是先得到SVM的解$(b_{svm},w_{svm})$，然后直接代入到logistic regression中，得到$g(x)=\theta(w_{svm}^Tx+b_{svm})$。<br>
这种方法直接使用了SVM和logistic regression的相似性，一般情况下表现还不错。<br>
但是，这种形式过于简单，与logistic regression的关联不大，没有使用到logistic regression中好的性质和方法。<br>
<br>
第二种简单的方法是同样先得到SVM的解$(b_{svm},w_{svm})$，然后把$(b_{svm},w_{svm})$作为logistic regression的初始值，再进行迭代训练修正，速度比较快<br>
最后，将得到的b和w代入到g(x)中。<br>
但并没有比直接使用logistic regression快捷多少。
![na](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/5_na.png?raw=true)<br>

***

构造一个融合两者优势的模型,我们额外增加了放缩因子A和平移因子B<br>
首先利用SVM的解$(b_{svm},w_{svm})$来构造这个模型，放缩因子A和平移因子B是待定系数。<br>
然后再用通用的logistic regression优化算法，通过迭代优化，得到最终的A和B。<br>
一般来说，如果$(b_{svm},w_{svm})$较为合理的话，满足A>0且$B\approx0$。<br>
![model](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/5_model.png?raw=true)<br>

***

得到了新的logistic regression:
![newlog](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/5_newlog.png?raw=true)<br>

***

其中的$(b_{svm},w_{svm})$已经在SVM中解出来了，实际上的未知参数只有A和B两个<br>
这种Probabilistic SVM的做法分为三个步骤：<br>
![psvm](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/5_psvm.png?raw=true)<br>

这种soft binary classifier方法得到的结果跟直接使用SVM classifier得到的结果可能不一样，这是因为我们引入了系数A和B<br>
一般来说，soft binary classifier效果更好<br>
logistic regression的解法，可以选择GD、SGD等等。

---

### 4.Kernel Logistic Regression
![z](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/5_z.png?raw=true)<br>
对于L2-regularized linear model，如果它的最小化问题形式为如下的话，那么最优解$w_*=\sum_{n=1}^N\beta_nz_n$。<br>
![l2](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/5_l2.png?raw=true)<br>

***

假如最优解$w_=w_{||}+w_{\bot}$。<br>
$w_{||}$和$w_{\bot}$分别是平行z空间和垂直z空间的部分。<br>
我们需要证明的是$w_{\bot}=0$。<br>
利用反证法，假如$w_{\bot}\neq0$，考虑$w_*$与$w_{||}$的比较。<br>
第一步先比较最小化问题的第二项：$err(y,w_*^Tz_n)=err(y_n,(w_{||}+w_{\bot})^Tz_n=err(y_n,w_{||}^Tz_n)$，即第二项是相等的。<br>
然后第二步比较第一项：$w_*^Tw_=w_{||}^Tw_{||}+2w_{||}^Tw_{\bot}+w_{\bot}^Tw_{\bot}>w_{||}^Tw_{||}$，即$w_*$对应的L2-regularized linear model值要比$w_{||}$大，这就说明$w_*$并不是最优解，从而证明$w_{\bot}$必然等于零，即$w_*=\sum_{n=1}^N\beta_nz_n$一定成立，$w_*$一定可以写成z的线性组合形式。<br>
![solve](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/5_solve.png?raw=true)<br>

***

将$w_=\sum_{n=1}^N\beta_nz_n$代入到L2-regularized logistic regression最小化问题中，得到：<br>
![solvel2](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/5_solvel2.png?raw=true)<br>

***

从另外一个角度来看Kernel Logistic Regression（KLR）：<br>
![klr](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/5_klr.png?raw=true)<br>
上式中log项里的$\sum_{m=1}^N\beta_mK(x_m,x_n)$可以看成是变量$\beta$和$K(x_m,x_n)$的内积。<br>
上式第一项中的$\sum_{n=1}^N\sum_{m=1}^N\beta_n\beta_mK(x_n,x_m)$可以看成是关于$\beta$的正则化项$\beta^TK\beta$。<br>
所以，KLR是$\beta$的线性组合，其中包含了kernel内积项和kernel regularizer。这与SVM是相似的形式。<br>
<br>
KLR中的$\beta_n$与SVM中的$\alpha_n$是有区别的。SVM中的$\alpha_n$大部分为零，SV的个数通常是比较少的；而KLR中的$\beta_n$通常都是非零值。