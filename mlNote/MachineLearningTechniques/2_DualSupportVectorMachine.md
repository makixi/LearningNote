# Dual Support Vector Machine

---

### 1.Motivation of Dual SVM
![svm](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/2_svmvs.png?raw=true)<br>
Original SVM二次规划问题的变量个数是$\hat d +1$，有N个限制条件；<br>
Equivalent SVM二次规划变量个数为N个，有N+1个限制条件。<br>
这种对偶SVM的好处就是问题只跟N有关，与$\hat d$无关，这样就不会出现当$\hat d$无限大时难以求解的情况。<br>

***

![lamda](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/2_lamda.png?raw=true)<br>
Regularization中，在最小化$E_{in}$的过程中，也添加了限制条件：$w^Tw\leq C$。我们的求解方法是引入拉格朗日因子$\lambda$<br>，将有条件的最小化问题转换为无条件的最小化问题：$min\ E_{aug}(w)=E_{in}(w)+\frac{\lambda}{N}w^Tw$，最终得到的w的最优化解为：<br>
<br>
$$\nabla E_{in}(w)+\frac{2\lambda}{N}w=0$$<br>
<br>
所以，在regularization问题中，$\lambda$是已知常量，求解过程变得容易。那么，对于dual SVM问题，同样可以引入$\lambda$，将条件问题转换为非条件问题，只不过$\lambda$是未知参数，且个数是N，需要对其进行求解。<br>

***

现在要将条件问题转化成非条件问题。<br>
SVM中，目标是：$min\ \frac12w^Tw$，条件是：$y_n(w^Tz_n+b)\geq 1,\ for\ n=1,2,\cdots,N$。首先，我们令拉格朗日因子为$\alpha_n$（区别于regularization），构造一个函数：<br>
<br>
$$L(b,w,\alpha)=\frac12w^Tw+\sum_{n=1}^N\alpha_n(1-y_n(w^Tz_n+b))$$
<br>
这个函数右边第一项是SVM的目标，第二项是SVM的条件和拉格朗日因子$\alpha_n$的乘积。我们把这个函数称为拉格朗日函数，其中包含三个参数：b，w，$\alpha_n$。<br>
![lagrange](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/2_lagrange.png?raw=true)<br>
<br>
再利用拉格朗日函数，把SVM构成一个非条件问题。<br>
![svm](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/2_svm.png?raw=true)<br>
首先我们规定拉格朗日因子$\alpha_n\geq0$，根据SVM的限定条件可得：$(1-y_n(w^Tz_n+b))\leq0$<br>
如果没有达到最优解，即有不满足$(1-y_n(w^Tz_n+b))\leq0$的情况，因为$\alpha_n\geq0$，那么必然有$\sum_n\alpha_n(1-y_n(w^Tz_n+b))\geq0$。<br>
对于这种大于零的情况，其最大值是无解的。<br>
<br>
如果对于所有的点，均满足$(1-y_n(w^Tz_n+b))\leq0$，那么必然有$\sum_n\alpha_n(1-y_n(w^Tz_n+b))\leq0$<br>
则当$\sum_n\alpha_n(1-y_n(w^Tz_n+b))=0$时，其有最大值，最大值就是我们SVM的目标：$\frac12w^Tw$。<br>
因此，这种转化为非条件的SVM构造函数的形式是可行的。

---


### 2.Lagrange Dual SVM 
现在SVM问题已经转化为与拉格朗日因子$\alpha_n$有关的最大最小值形式。已知$\alpha_n\geq0$，那么对于任何固定的$\alpha'$，且$\alpha_n'\geq0$，一定有如下不等式成立：<br>
![不等式](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/2_maxmin.png?raw=true)<br>
<br>
对上述不等式右边取max，不等式依然成立。<br>
![不等式](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/2_maxmin2.png?raw=true)<br>
<br>
上述不等式表明，我们对SVM的min和max做了对调，满足这样的关系，这叫做Lagrange dual problem。不等式右边是SVM问题的下界，我们接下来的目的就是求出这个下界。<br>

***

已知$\geq$是一种弱对偶关系，在二次规划QP问题中，如果满足以下三个条件:<br>
1.函数是凸的（convex primal）<br>
2.函数有解（feasible primal）<br>
3.条件是线性的（linear constraints）<br>

那么，上述不等式关系就变成强对偶关系，$\geq$变成=，即一定存在满足条件的解$(b,w,\alpha)$，使等式左边和右边都成立，SVM的解就转化为右边的形式。<br>
<br>
经过推导，SVM对偶问题的解已经转化为无条件形式：<br>
![dual_svm](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/2_dualsvm.png?raw=true)<br>
上式括号里面的是对拉格朗日函数$L(b,w,\alpha)$计算最小值。<br>
那么根据梯度下降算法思想:最小值位置满足梯度为零。<br>
首先，令$L(b,w,\alpha)$对参数b的梯度为零:

$$\frac{\partial L(b,w,\alpha)}{\partial b}=0=-\sum_{n=1}^N\alpha_ny_n$$

那么，我们把这个条件代入计算max条件中（与$\alpha_n\geq0$同为条件），并进行化简：<br>
![max_cal](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/2_maxcal.png?raw=true)<br>
<br>
这样，SVM表达式成功消去了b。<br>
现在，令$L(b,w,\alpha)$对参数w的梯度为零:

$$\frac{\partial L(b,w,\alpha)}{\partial w}=0=w-\sum_{n=1}^N\alpha_ny_nz_n$$

同样把这个条件带入原式并化简：<br>
![max_cal2](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/2_maxcal2.png?raw=true)<br>

***

这样，SVM表达式成功消去了w。问题更加简化，这时候的条件有三个：<br>
1. all $\alpha_n\geq0$<br>
2. $\sum_{n=1}^N\alpha_ny_n=0$<br>
3. $w=\sum_{n=1}^N\alpha_ny_nz_n$<br>
<br>
SVM简化为只有$\alpha_n$的最佳化问题，即计算满足上述三个条件下，函数$-\frac12||\sum_{n=1}^N\alpha_ny_nz_n||^2+\sum_{n=1}^N\alpha_n$最小值时对应的$\alpha_n$是多少。<br>

总结一下，SVM最佳化形式转化为只与$\alpha_n$有关：<br>

![svm_best](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/2_svmbest.png?raw=true)<br>
<br>
其中，满足最佳化的条件称之为Karush-Kuhn-Tucker(KKT)：<br>
![kkt](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/2_kkt.png?raw=true)<br>


---

### 3.Solving Dual SVM 
将max问题转化为min问题，再做一些条件整理和推导。<br>
![convexQP](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/2_convexQP.png?raw=true)<br>


![solveQP](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/2_solveQP.png?raw=true)<br>
显然，这是一个convex的QP问题，且有N个变量$\alpha_n$，限制条件有N+1个。用QP解法，找到Q，p，A，c对应的值，用软件工具包进行求解即可。<br>
<br>

***

![note](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/2_note.png?raw=true)<br>
$q_{n,m}=y_ny_mz^T_nz_m$，大部分值是非零的，称为dense。<br>
当N很大的时候，那么对应的$Q_D$的计算量将会很大，存储空间也很大。<br>

***

![solveKKT](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/2_solveKKT.png?raw=true)<br>
得到$\alpha_n$之后，再根据之前的KKT条件，就可以计算出w和b了。<br>
首先利用条件$w=\sum\alpha_ny_nz_n$得到w，然后利用条件$\alpha_n(1-y_n(w^Tz_n+b))=0$，取任一$\alpha_n\neq0$即$\alpha_n$>0的点，得到$1-y_n(w^Tz_n+b)=0$<br>
进而求得$b=y_n-w^Tz_n$。<br>

***

值得注意的是，计算b值，$\alpha_n$>0时，有$y_n(w^Tz_n+b)=1$成立。$y_n(w^Tz_n+b)=1$正好表示的是该点在SVM分类线上，即fat boundary。也就是说，满足$\alpha_n$>0的点一定落在fat boundary上，这些点就是Support Vector。这是一个非常有趣的特性。

---

### 4.Messages behind Dual SVM 
![message](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/2_message.png?raw=true)<br>
把位于分类线边界上的点称为support vector（candidates）。<br>
$\alpha_n$>0的点一定落在分类线边界上，这些点称之为support vector<br>
也就是说分类线上的点不一定都是支持向量，但是满足$\alpha_n$>0的点，一定是支持向量。<br>
![sv](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/2_sv.png?raw=true)<br>

***

![svm_vs_pla](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/2_svmpla.png?raw=true)<br>
$w_{SVM}$由fattest hyperplane边界上所有的SV决定，$w_{PLA}$由所有当前分类错误的点决定。<br>
$w_{SVM}$和$w_{PLA}$都是原始数据点$y_nz_n$的线性组合形式，是原始数据的代表。<br>

***

![lastvs](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/2_lastvs.png?raw=true)<br>
总结一下，本节课和上节课主要介绍了两种形式的SVM<br>
一种是Primal Hard-Margin SVM，另一种是Dual Hard_Margin SVM。Primal Hard-Margin <br>SVM有$\hat d+1$个参数，有N个限制条件。当$\hat d+1$很大时，求解困难。<br>
而Dual Hard_Margin SVM有N个参数，有N+1个限制条件。当数据量N很大时，也同样会增大计算难度。<br>
两种形式都能得到w和b，求得fattest hyperplane。通常情况下，如果N不是很大，一般使用Dual SVM来解决问题。<br>


### 总结
本节课主要介绍了SVM的另一种形式：Dual SVM。我们这样做的出发点是为了移除计算过程对$\hat d$的依赖。Dual SVM的推导过程是通过引入拉格朗日因子$\alpha$，将SVM转化为新的非条件形式。然后，利用QP，得到最佳解的拉格朗日因子$\alpha$。再通过KKT条件，计算得到对应的w和b。最终求得fattest hyperplane。下一节课，我们将解决Dual SVM计算过程中对$\hat d$的依赖问题。

