## HW9

#### Q1

略

#### Q2

例如某企业进行市场调研，抽样调查其产品受欢迎程度时，选择了几家大型商场统计近一个月以来的销售情况。但这样可能忽略其他小型商场的销售情况、消费者的年龄性别等特征以及产品受季度等因素的影响。

#### Q3

(1) 证明：对于无放回抽样，每次抽取到同一元素的概率都相等，因此 $E(X_i)=\mu,Var(X_i)=\sigma^2$

(2) 证明：$\forall 1\le i,j\le n$ 且 $i\ne j$，有 $Cov(X_i,X_j)=E(X_iX_j)-E(X_i)E(X_j)=\left(\mu^2-\dfrac{\sigma^2}{N-1}-\mu^2\right)=-\dfrac{\sigma^2}{N-1}$

于是 $E(\bar X)=\dfrac1n\sum_{i=1}^nE(X_i)=\dfrac1n\cdot n\mu = \mu $

于是 $Var(\bar X)=\dfrac1 {n^2} Var(\sum_{i=1}^nX_i)=\dfrac1 {n^2} (nVar(X_i)+\sum_{i\ne j}Cov(X_i,X_j))=\dfrac1 {n^2}(n\sigma^2-n(n-1)\dfrac{\sigma^2}{N-1})=\dfrac{\sigma^2}{n}\left(\dfrac{N-n}{N-1}\right)$

#### Q4

(1) 根据题意，$\mu_1\approx kp$，$m_2\approx kp(1-p)$，于是 $\hat k=\dfrac{\mu_1^2}{\mu_1-m_2},\hat p=1-\dfrac{m_2}{\mu_1}$

(2) 样本量较少时，可能出现 $m_2\ge\mu_1$，此时 $\hat k,\hat p\le 0$，估计不再合理。

#### Q5

对于矩估计，$\dfrac32\theta\approx\bar X,\hat\theta=\dfrac23\bar X$

对于 MLE，$L(\theta)=\dfrac1{\theta^n},\theta\lt X_i\lt2\theta$，故 $\theta^*=\dfrac12\max\{X_1,\cdots,X_n\}$

#### Q6

(1) 证明：首先，显然有 $f(x;a,\sigma)\gt0$，其次 $\int_{\mathbb R}f(x;a,\sigma)dx=\dfrac1{\sqrt{2\pi}\sigma^3}(x-a)^2e^{-\tfrac{(x-a)^2}{2\sigma^2}}dx=1$

因此 $f(x;a,\sigma)$ 满足概率密度的归一化要求。

(2) $\bar X\approx\int_{\mathbb R}xf(x;a,\sigma)dx=\int_{\mathbb R}(2a-x)f(x;a,\sigma)dx=2a-\int_{\mathbb R}xf(x;a,\sigma)dx$，因此 $\hat a\approx\bar X$，于是

$$
\begin{align}
m_2\approx&\int_{\mathbb R}(x-a)^2f(x;a,\sigma)dx\\
&=\dfrac1{\sqrt{2\pi}\sigma^3}\int_{\mathbb R}x^4e^{-\tfrac{x^2}{2\sigma^2}}dx\\
&\xlongequal{x=\sqrt2\sigma y}\dfrac{4\sigma^2}{\sqrt\pi}\int_{\mathbb R}y^4e^{-y^2}dy\\
&=\dfrac{4\sigma^2}{\sqrt\pi}(\dfrac12y^3e^{-y^2}\bigg|_{-\infty}^{+\infty}+\dfrac32\int_{\mathbb R}y^2e^{-y^2}dy)\\
&=\dfrac{6\sigma^2}{\sqrt\pi}\int_{\mathbb R}y^2e^{-y^2}dy\\
&=\dfrac{6\sigma^2}{\sqrt\pi}(-\dfrac12ye^{-y^2}\bigg|_{-\infty}^{+\infty}+\dfrac12\int_{\mathbb R}e^{-y^2}dy)\\
&=3\sigma^2
\end{align}
$$

因此 $\hat{\sigma^2}=\dfrac13m^2$

(3) $L(\theta)=\prod_{i=1}^{n}f(X_i;a,\sigma)={(\sqrt{2\pi}\sigma^3)}^{-n}\prod_{i=1}^{n}(x_i-a)^2e^{-\tfrac{\sum_{i=1}^{n}(X_i-a)^2}{2\sigma^2}}$

需要满足方程 $\left\{\begin{align}\dfrac{\partial(\ln L)}{\partial a}=0\\\dfrac{\partial(\ln L)}{\partial a}=0\end{align}\right.$，可使用牛顿—拉夫逊迭代法求解。

#### Q7

对于矩估计，$\bar X\approx p$，故 $\hat p=\bar X$

对于 MLE，有 $L(p)=\prod_{i=1}^np^{X_i}{(1-p)}^{1-X_i}=p^{n\bar X}{(1-p)}^{n(1-\bar X)}$

令 $\dfrac{\partial(\log L)}{\partial p}=n\bar X\cdot\dfrac1p+n(1-\bar X)\dfrac1{p-1}=0$ ，解得 $p^*=\bar X$

#### Q8

根据题意，$L(p_1,\cdots,p_n)=p_1^{X_1}p_2^{X_2}\cdots p_m^{X_m}$，故 $\log L(p_1,\cdots,p_n)=\sum_{i=1}^mX_i\ln p_i$

由于 $p_1+\cdots+p_n=1,X_1+\cdots+X_m=n$，由拉格朗日乘子法，令

$F(p_1,\cdots,p_m,\lambda,\mu)=\sum_{i=1}^mX_i\ln p_i+\lambda(p_1+\cdots+p_n-1)+\mu(X_1+\cdots+X_m-n)$

则 $\left\{\begin{align}&\dfrac{\partial F}{\partial {p_i}}=\dfrac{X_i}{p_i}+\lambda=0，i=1,\cdots,m\\&\dfrac{\partial F}{\partial \lambda}=p_1+\cdots+p_n-1=0\\&\dfrac{\partial F}{\partial \mu}=X_1+\cdots+X_m-n=0 \end{align}\right.$，解得 $p_i^*=\dfrac{X_i}n,i=1,2,\cdots,m$

#### Q9

对于矩估计，$\bar X\approx E(X)=2\theta^2-4\theta+3$，所以 $2\hat\theta^2-4\hat\theta+3=\dfrac43$，解得 $\hat\theta=\dfrac56$

对于 MLE，$L(\theta)=\theta^2(2\theta(1-\theta))\cdot\theta^2=2\theta^5(1-\theta)$

令 $\dfrac{\partial L}{\partial \theta}=10\theta^4-12\theta^5=0$，解得 $\theta^*=\dfrac56$

#### Q10

(1) 对于矩估计，$\bar X\approx E(X)=\int_0^1xf(x)dx=\dfrac{\theta}{\theta+1}$，因此 $\hat\theta=\dfrac{\bar X}{1-\bar X}$

(2) $L(\theta)=\prod_{i=1}^{n}f(X_i)=\theta^n(\sum_{i=1}^nX_i)^{\theta-1}$

令 $\dfrac{\partial(\log L)}{\partial\theta}=\dfrac n\theta+\sum_{i=1}^n\log X_i=0$，解得 $\theta^*=-\dfrac{n}{\sum_{i=1}^n\log X_i}$

#### Q11

代码如下：

```python
import numpy as np

n = int(input("n ="))
k = int(input("k ="))
p = float(input("p ="))

rdm = np.random.binomial(k, p, n)

mean = np.mean(rdm)
var = np.var(rdm)

k_ = (mean**2) / (mean - var)
p_ = 1 - var / mean

print(f"When n = {n}, k = {k}, p = {p}")
print(f"mean = {mean}, var = {var}")
print(f"MM: k = {k_}, p = {p_}")
```

该代码会给出提示，接受 $n,k,p$ 三个参数的输入，并给出矩估计。结果如下：

当 $k = 10, p =  0.01,n=10$

|   Attempts   |  1   |       2        |  3   |  4   |       5        |
| :----------: | :--: | :------------: | :--: | :--: | :------------: |
| k_estimation |  1   | nan (mean=var) |  1   |  1   | nan (mean=var) |
| p_estimation | 0.2  |  nan (mean=0)  | 0.2  | 0.1  |  nan (mean=0)  |
|  明显不合理  |  *   |       *        |  *   |  *   |       *        |

当 $k = 10, p =  0.01,n=1000$

|   Attempts   |   1   |   2   |   3   |   4   |   5    |
| :----------: | :---: | :---: | :---: | :---: | :----: |
| k_estimation | 6.68  | 1.17  | 4.63  | 13.25 | 10.76  |
| p_estimation | 0.015 | 0.099 | 0.022 | 0.008 | 0.0098 |
|  明显不合理  |       |   *   |   *   |       |        |

当 $k = 10, p =  0.5,n=10$

|   Attempts   |  1   |  2   |  3   |  4   |   5   |
| :----------: | :--: | :--: | :--: | :--: | :---: |
| k_estimation | 5.98 | 7.36 | 6.58 | 7.64 | 14.37 |
| p_estimation | 0.72 | 0.73 | 0.76 | 0.61 | 0.35  |
|  明显不合理  |      |      |      |      |       |

当 $k = 10, p =  0.5,n=1000$

|   Attempts   |   1   |  2   |   3   |  4   |  5   |
| :----------: | :---: | :--: | :---: | :--: | :--: |
| k_estimation | 10.28 | 9.66 | 10.14 | 9.81 | 9.92 |
| p_estimation | 0.49  | 0.52 | 0.49  | 0.51 | 0.51 |
|  明显不合理  |       |      |       |      |      |

分析上面的表格，可以发现，最后一组的矩估计最为精确，说明要使矩估计尽可能精确，二项分布不应该过于“极端”，即 $p$ 应该尽量接近 $0.5$，同时，样本容量 $n$ 不宜过小。
