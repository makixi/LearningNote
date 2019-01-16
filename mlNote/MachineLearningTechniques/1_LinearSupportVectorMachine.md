# Linear Support Vector Machine

---

### 1.Large-Margin Separating Hyperplane
![3lines](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/1_3lines.png?raw=true)<br>
上图三种分割方式都可以正确分割正负点，那么怎么分辨哪种方案更好？<br>
<br>
PLA可能会随机选择方案(最终结果与经过的错误点有关)<br>
都满足VC bound要求，模型复杂度一样。<br>

***

![3answers](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/1_3answers.png?raw=true)<br>
每个样本点距离分界线越远，就表明其对于测量误差的容忍度越高，就越安全。（PS:测量误差是一种类型的noise，而noise是导致过拟合的一个原因）<br>
![bestline](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/1_bestline.png?raw=true)<br>

***
分类线由权重w决定，目的就是找到使margin最大时对应的w值。即<br>
![bestmargin](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/1_maxmargin.png?raw=true)<br>

---

### 2.Standard Large-Margin Problem
如何计算点到直线的距离？<br>
首先，我们将权重$w(w_0,w_1,\cdots,w_d)$中的$w_0$拿出来，用b表示(即截距)。同时省去$x_0$项。这样，hypothesis就变成了$h(x)=sign(w^Tx+b)$。<br>
![shorten_X_and_W](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/1_shortenXW.png?raw=true)<br>
![distances](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/1_distances.png?raw=true)<br>
目标形式转换为：<br>
![goaldistances](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/1_goaldistance.png?raw=true)<br>
进行简化：<br>
![simplygoal](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/1_simplygoal.png?raw=true)<br>
我们的目标就是根据这个条件，计算$\frac1{||w||}$的最大值。<br>

***

可以把目标$\frac1{||w||}$最大化转化为计算$\frac12w^Tw$的最小化问题<br>
![finalgoal](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/1_finalgoal.png?raw=true)<br>
最终的条件就是$y_n(w^Tx_n+b)\geq 1$，而我们的目标就是最小化$\frac12w^Tw$值。

---

### 3.Support Vector Machine 
现在，条件和目标变成：<br>
![3initgoal](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/1_3initgoal.png?raw=true)<br>

Support Vector Machine(SVM)这个名字从何而来？为什么把这种分类面解法称为支持向量机呢？这是因为分类面仅仅由分类面的两边距离它最近的几个点决定的，其它点对分类面没有影响。决定分类面的几个点称之为支持向量（Support Vector），好比这些点“支撑”着分类面。而利用Support Vector得到最佳分类面的方法，称之为支持向量机（Support Vector Machine）。<br>

***

这是一个典型的二次规划问题，即Quadratic Programming（QP）。因为SVM的目标是关于w的二次函数，条件是关于w和b的一次函数，所以，它的求解过程还是比较容易的，可以使用一些软件（例如Matlab）自带的二次规划的库函数来求解。下图给出SVM与标准二次规划问题的参数对应关系：<br>
![QP](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/1_QP.png?raw=true)<br>

***

那么，**线性SVM**算法可以总结为三步：<br>
1.计算对应的二次规划参数Q，p，A，c<br>
2.根据二次规划库函数，计算b，w<br>
3.将b和w代入$g_{SVM}$，得到最佳分类面<br>

![svmstep](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/1_svmstep.png?raw=true)<br>
如果是非线性的，可以先用特征转换的方法，先做特征变换。将非线性的x域映射到线性的z域，再利用线性SVM算法进行求解。

---


### 4.Reasons behind Large-Margin Hyperplane
SVM的思想与正则化regularization思想很类似。<br>
![svm_vs_regularization](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/1_svmregu.png?raw=true)<br>

***

如果Dichotomies越少，那么复杂度就越低，即有效的VC Dimension就越小，得到$E_{out}\approx E_{in}$，泛化能力强。

---

### 总结
本节课主要介绍了线性支持向量机（Linear Support Vector Machine）。我们先从视觉角度出发，希望得到一个比较“胖”的分类面，即满足所有的点距离分类面都尽可能远。<br>
然后，我们通过一步步推导和简化，最终把这个问题转换为标准的二次规划（QP）问题。二次规划问题可以使用软件来进行求解，得到我们要求的w和b，确定分类面。<br>
这种方法背后的原理其实就是减少了dichotomies的种类，减少了有效的VC Dimension数量，从而让机器学习的模型具有更好的泛化能力。