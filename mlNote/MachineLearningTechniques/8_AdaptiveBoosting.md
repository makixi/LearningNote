# Adaptive Boosting

---

### 1.Motivation of Boosting 
20张图片包括它的标签都是已知的。<br>
根据苹果是圆形的这个判断，大部分苹果能被识别，但是也存在错误。<br>
![1](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/8_1.png?raw=true)<br>
蓝色区域代表分类错误。<br>
<br>
把蓝色区域（分类错误的图片）放大，分类正确的图片缩小，这样在接下来的分类中就会更加注重这些错误样本。<br>

***

根据苹果是红色这个信息，得到的结果（蓝色区域代表分类错误）：<br>
![2](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/8_2.png?raw=true)<br>
然后将分类错误的样本放大化，其它正确的样本缩小化。

---

苹果也可能是绿色的<br>
![3](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/8_3.png?raw=true)<br>
蓝色区域的图片代表分类错误<br>
把这些分类错误的样本放大化，其它正确的样本缩小化，在下一轮判断继续对其修正。<br>

***

上面有梗的才是苹果<br>
![4](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/8_4.png?raw=true)<br>
苹果被定义为：圆的，红色的，也可能是绿色的，上面有梗。从一个一个的推导过程中，我们似乎得到一个较为准确的苹果的定义。虽然可能不是非常准确，但是要比单一的条件要好得多。<br>
简单的hypotheses $g_t$，将所有$g_t$融合，得到很好的预测模型G。<br>
![5](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/8_5.png?raw=true)<br>

***

![6](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/8_6.png?raw=true)<br>
不同的判断代表不同的hypotheses $g_t$<br>
最终得到的结果定义就代表hypothesis G<br>
演算法将注意力集中到错误样本，从而得到更好的定义。

---

### 2.Diversity by Re-weighting
Bagging的核心是bootstrapping，通过对原始数据集D不断进行bootstrap的抽样动作，得到与D类似的数据集$\hat{D}_t$，每组$\hat{D}_t$都能得到相应的$g_t$，从而进行aggregation的操作。<br>
假如包含四个样本的D经过bootstrap，得到新的$\hat{D}_t$如下：<br>
![7](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/8_7.png?raw=true)<br>

***

那么，对于新的$\hat{D}t$，把它交给base algorithm，找出$E{in}$最小时对应的$g_t$，如下图右边所示。

$$E_{in}^{0/1}(h)=\frac14\sum_{n=1}^4[y\neq h(x)]$$

由于$\hat{D}_t$完全是D经过bootstrap得到的，其中样本$(x_1,y_1)$出现2次，$(x_2,y_2)$出现1次，$(x_3,y_3)$出现0次，$(x_4,y_4)$出现1次。引入一个参数$u_i$来表示原D中第i个样本在$\hat{D}_t$中出现的次数，如下图左边所示。

$$E_{in}^u(h)=\frac14\sum_{n=1}^4u_n^{(t)}\cdot [y_n\neq h(x)]$$

![8](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/8_8.png?raw=true)<br>
参数u相当于是权重因子，当$\hat{D}_t$中第i个样本出现的次数越多的时候，那么对应的$u_i$越大，表示在error function中对该样本的惩罚越多。<br>

![9](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/8_9.png?raw=true)<br>

***

在logistic regression中，同样可以对每个犯错误的样本乘以相应的$u_n$，作为惩罚因子。$u_n$表示该错误点出现的次数，$u_n$越大，则对应的惩罚因子越大，则在最小化error时就应该更加重视这些点。<br>
![10](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/8_10.png?raw=true)<br>

***

$g_t$越不一样，其aggregation的效果越好<br>
![11](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/8_11.png?raw=true)<br>
利用$g_t$在使用$u_n^{(t+1)}$的时候表现很差的条件，越差越好。<br>
这样的做法就能最大限度地保证$g_{t+1}$会与$g_t$有较大的差异性。<br>
![12](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/8_12.png?raw=true)<br>

分式中分子可以看成$g_t$作用下犯错误的点，而分母可以看成犯错的点和没有犯错误的点的集合，即所有样本点。<br>
其中犯错误的点和没有犯错误的点分别用橘色方块和绿色圆圈表示：<br>

![13](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/8_13.png?raw=true)<br>
在$g_t$作用下，让犯错的$u_n^{(t+1)}$数量和没有犯错的$u_n^{(t+1)}$数量一致（包含权重$u_n^{t+1}$）就可以使分式结果为）0.5。<br>
一种简单的方法就是利用放大和缩小的思想（本节课开始引入识别苹果的例子中提到的放大图片和缩小图片就是这个目的），将犯错误的$u_n^{t}$和没有犯错误的$u_n^{t}$做相应的乘积操作，使得二者值变成相等。<br>

***

或者利用犯错的比例来做。<br>
一般求解方式是令犯错率为$\epsilon_t$，在计算$u_n^{(t+1)}$的时候，$u_n^{t}$分别乘以$(1-\epsilon_t)$和$\epsilon_t$。<br>
![14](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/8_14.png?raw=true)<br>

---

### 3.Adaptive Boosting Algorithm
新的尺度因子：

$$\diamond t=\sqrt{\frac{1-\epsilon_t}{\epsilon_t}}$$

对错的$u_n^{t}$，将它乘以$\diamond t$；对于正确的$u_n^{t}$，将它除以$\diamond t$。<br>
这种操作跟之前介绍的分别乘以$(1-\epsilon_t)$和$\epsilon_t$的效果是一样的。<br>
<br>
如果$\epsilon_t\leq\frac12$，得到$\diamond t\geq1$，那么接下来错误的$u_n^{t}$与$\diamond t$的乘积就相当于把错误点放大了，而正确的$u_n^{t}$与$\diamond t$的相除就相当于把正确点缩小了。<br>
也就是能够将注意力更多地放在犯错误的点上。<br>
通过这种scaling-up incorrect的操作，能够保证得到不同于$g_t$的$g_{t+1}$。<br>

![15](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/8_15.png?raw=true)<br>

***

得到一个初步的演算法。其核心步骤是每次迭代时，利用$\diamond t=\sqrt{\frac{1-\epsilon_t}{\epsilon_t}}$把$u_t$更新为$u_{t+1}$。<br>
具体迭代步骤如下：<br>
![16](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/8_16.png?raw=true)<br>
一般来说，为了保证第一次$E_{in}$最小的话，设$u^{(1)}=\frac1N$。<br>
对所有的g(t)进行linear或者non-linear组合来得到G(t)。<br>
![17](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/8_17.png?raw=true)<br>

***

将所有的g(t)进行linear组合。<br>
方法是计算$g(t)$的同时，就能计算得到其线性组合系数$\alpha_t$，即aggregate linearly on the fly。<br>
这种算法使最终求得$g_{t+1}$的时候，所有$g_t$的线性组合系数$\alpha$也求得了，不用再重新计算$\alpha$了。<br>
这种Linear Aggregation on the Fly算法流程为：<br>
![18](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/8_18.png?raw=true)<br>

$\alpha_t$与$\epsilon_t$是相关的：$\epsilon_t$越小，对应的$\alpha_t$应该越大，$\epsilon_t$越大，对应的$\alpha_t$应该越小。<br>
又因为$\diamond t$与$\epsilon_t$是正相关的，所以，$\alpha_t$应该是$\diamond t$的单调函数。<br>
我们构造$\alpha_t$为：

$$\alpha_t=ln(\diamond t)$$

$\alpha_t$这样取值是有物理意义的<br>
例如当$\epsilon_t=\frac12$时，error很大，跟随机过程没什么两样，此时对应的$\diamond t=1$，$\alpha_t=0$，即此$g_t$对G没有什么贡献，权重应该设为零。<br>
而当$\epsilon_t=0$时，没有error，表示该$g_t$预测非常准，此时对应的$\diamond t=\infty$，$\alpha_t=\infty$，即此$g_t$对G贡献非常大，权重应该设为无穷大。<br>
![19](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/8_19.png?raw=true)<br>
这种算法被称为Adaptive Boosting。它由三部分构成：base learning algorithm A，re-weighting factor $\diamond t$和linear aggregation $\alpha_t$。这三部分分别对应于我们在本节课开始介绍的例子中的Student，Teacher和Class。
![20](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/8_20.png?raw=true)<br>

***

综上所述，完整的adaptive boosting（AdaBoost）Algorithm流程如下：
![21](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/8_21.png?raw=true)<br>
<br>
从VC bound角度来看，AdaBoost算法理论上满足：<br>
![22](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/8_22.png?raw=true)<br>

***

只要每次的$\epsilon_t\leq \epsilon<\frac12$，即所选择的矩g比乱猜的表现好一点点，那么经过每次迭代之后，矩g的表现都会比原来更好一些，逐渐变强，最终得到$E_{in}=0$且$E_{out}$很小。<br>
![23](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/8_23.png?raw=true)<br>

---

### 4.Adaptive Boosting in Action
eg:AdaBoost使用decision stump解决实际问题:<br>
二维平面上分布一些正负样本点，利用decision stump来做切割。<br>
![24](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/8_24.png?raw=true)<br>
<br>
第一步:<br>
![25](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/8_25.png?raw=true)<br>
<br>
第二步：<br>
![26](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/8_26.png?raw=true)<br>
<br>
第三步：<br>
![27](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/8_27.png?raw=true)<br>
<br>
第四步：<br>
![28](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/8_28.png?raw=true)<br>
<br>
第五步：<br>
![29](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/8_29.png?raw=true)<br>
<br>
经过5次迭代之后，所有的正负点已经被完全分开了，则最终得到的分类线为：<br>
![30](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/8_30.png?raw=true)<br>

***

对于一个相对比较复杂的数据集，如下图所示。它的分界线从视觉上看应该是一个sin波的形式。如果我们再使用AdaBoost算法，通过decision stump来做切割。在迭代切割100次后，得到的分界线如下所示。<br>
![31](https://github.com/makixi/MachineLearningNote/blob/master/MachineLearningTechniques/pic/8_31.png?raw=true)<br>
AdaBoost-Stump这种非线性模型得到的分界线对正负样本有较好的分离效果。<br>

### 5.Summary 
主要介绍了Adaptive Boosting。<br>
Boosting的思想，即把许多“弱弱”的hypotheses合并起来，变成很强的预测模型。<br>
算法如何实现，关键在于每次迭代时，给予样本不同的系数u，宗旨是放大错误样本，缩小正确样本，得到不同的小矩g。<br>
并且在每次迭代时根据错误$\epsilon$值的大小，给予不同$g_t$不同的权重。<br>
最终由不同的$g_t$进行组合得到整体的预测模型G。<br>
实际证明，Adaptive Boosting能够得到有效的预测模型。