## HW1

#### Q1

概率论起源于赌博中的 $de \ Méré’s \ Problem$，赌徒关心将赌注下在多少点最有利，后来这一问题从赌场迁移到了数学界，从大量赌博经验的积累到理论上的推导证明，概率论起源于实际问题，又以一种科学的视角为实际问题提供了可靠的解决方案。事实上，很多科学学科的建立过程都是如此，它们**来源**于实际问题，又**指导**了现实问题，即科学并非凭空产生，而**与现实生活紧密相关**。

**概率论与其他数学分支密不可分**。例如，20世纪初建立起的抽象测度和积分理论，为概率公理体系的建立奠定了基础。而 $Kolmogorov$ 运用公理化的方法，让概率论成为严谨的数学分支。而伊藤清则在概率论的基础上引进了随机积分和随机微分方程，为随机分析的建立奠定了基础。由此可见，概率论与其他数学分支密不可分，是数学大厦不可或缺的组成部分。

#### Q2

$Bertrand$ 悖论源于其描述中**“随机”**一词的模糊性，这启示我们在研究数学问题过程中，应当**注意问题的描述是否严谨且无歧义**。另一方面，$Bertrand$ 悖论的出现促进了概率论公理化的发展，现代概率论的公理化体系被提出，我认为这反映了一种趋势，即为使某个数学分支的研究更加**系统化**和**规范化**，使它的发展更加**稳健**且**可持续**，**公理化方法**的引进是必要的。

#### Q3

甲150元，乙50元比较合理，理由如下：

“甲乙两人水平相当”可理解为每人每局获胜概率相等，均为0.5。**假设**比赛继续进行下去，则乙要想最终获胜，必须连赢两局，其概率为$0.5 \times 0.5 = 0.25$，而甲为 $0.75$。而奖金分配应与获胜概率成正比，因此甲、乙两人应按照 $0.75:0.25=3:1$ 的比例分配奖金。

#### Q4

假设此人对A队获胜的主管概率为 $p$，根据$主观最终受益期望 \geq 开局投资量$，有：

$$
(20+5) \times p +0 \times (1-p) \geq 20
$$

解得 $p \geq 0.8$，故主观概率至少为 $0.8$。

#### Q5

投注人角度，每匹马获胜概率应与下注量成正比，故：

$$
A马获胜概率=\frac{500}{500+300+200}=0.5
$$

$$
B马获胜概率=\frac{300}{500+300+200}=0.3
$$

$$
C马获胜概率=\frac{200}{500+300+200}=0.2
$$

#### Q6

* 假想情况如下所示：

| 次数 $n$ | H或T | 频数 $n(H)$ | 相对频数 $n(H)/n$ |
| :------: | :--: | :---------: | :---------------: |
|1|H|1|1.0000|
|2|H|2|1.0000|
|3|T|2|0.6667|
|4|H|3|0.7500|
|5|T|3|0.6000|
|6|H|4|0.6667|
|7|T|4|0.5714|
|8|T|4|0.5000|
|9|T|4|0.4444|
|10|T|4|0.4000|
|11|T|4|0.3636|
|12|T|4|0.3333|
|13|T|4|0.3077|
|14|H|5|0.3571|
|15|T|5|0.3333|
|16|H|6|0.3750|
|17|T|6|0.3529|
|18|H|7|0.3889|
|19|T|7|0.3684|
|20|H|8|0.4000|
|21|H|9|0.4286|
|22|H|10|0.4545|
|23|H|11|0.4783|
|24|H|12|0.5000|
|25|T|12|0.4800|
|26|H|13|0.5000|
|27|H|14|0.5185|
|28|H|15|0.5357|
|29|H|16|0.5517|
|30|H|17|0.5667|
|31|T|17|0.5484|
|32|T|17|0.5312|
|33|H|18|0.5455|
|34|T|18|0.5294|
|35|T|18|0.5143|
|36|H|19|0.5278|
|37|T|19|0.5135|
|38|T|19|0.5000|
|39|T|19|0.4872|
|40|H|20|0.5000|
|41|H|21|0.5122|
|42|H|22|0.5238|
|43|T|22|0.5116|
|44|H|23|0.5227|
|45|T|23|0.5111|
|46|T|23|0.5000|
|47|H|24|0.5106|
|48|T|24|0.5000|
|49|H|25|0.5102|
|50|T|25|0.5000|

* 实际操作如下所示：

| 次数 $n$ | H或T | 频数 $n(H)$ | 相对频数 $n(H)/n$ |
| :------: | :--: | :---------: | :---------------: |
|1|T|0|0.0000|
|2|H|1|0.5000|
|3|H|2|0.6667|
|4|T|2|0.5000|
|5|T|2|0.4000|
|6|T|2|0.3333|
|7|H|3|0.4286|
|8|T|3|0.3750|
|9|H|4|0.4444|
|10|H|5|0.5000|
|11|T|5|0.4545|
|12|T|5|0.4167|
|13|T|5|0.3846|
|14|T|5|0.3571|
|15|H|6|0.4000|
|16|H|7|0.4375|
|17|T|7|0.4118|
|18|T|7|0.3889|
|19|T|7|0.3684|
|20|H|8|0.4000|
|21|H|9|0.4286|
|22|T|9|0.4091|
|23|T|9|0.3913|
|24|H|10|0.4167|
|25|H|11|0.4400|
|26|T|11|0.4231|
|27|T|11|0.4074|
|28|T|11|0.3929|
|29|T|11|0.3793|
|30|H|12|0.4000|
|31|H|13|0.4194|
|32|T|13|0.4062|
|33|T|13|0.3939|
|34|T|13|0.3824|
|35|H|14|0.4000|
|36|T|14|0.3889|
|37|T|14|0.3784|
|38|T|14|0.3684|
|39|H|15|0.3846|
|40|H|16|0.4000|
|41|H|17|0.4146|
|42|T|17|0.4048|
|43|T|17|0.3953|
|44|H|18|0.4091|
|45|H|19|0.4222|
|46|H|20|0.4348|
|47|H|21|0.4468|
|48|H|22|0.4583|
|49|H|23|0.4694|
|50|T|23|0.4600|

我选择的是一元硬币，上述结果表明，实际操作与假想实验有相同的趋势，即随着抛硬币次数的增加，相对频数显现稳定趋势，趋近于某个特定的值。

在假想试验中，相对频数趋于 $0.5$，但实际试验结果不一定趋近该值。欲比较两者是否存在实质性差异，需要**足够多的**试验次数，使相对频数真正稳定下来（即波动范围控制在某个很小的阈值以内）。而实际操作中，根据上述**统计**的原理比较差异的做法是难以实现的，因此可以检查硬币正反两面形状的差异，落点所在平面是否平整，实验者的抛硬币特点等因素是否存在差异，这些都会影响假想试验与实际试验结果的实质性差异。

#### Q7

$$
A=\{(0,s),(1,s)\}
$$

$$
B=\{(0,g),(0,f),(0,s)\}
$$

$$
B^c+A=\{(0,s),(1,g),(1,f),(1,s)\}
$$

#### Q8

证明：根据事件的运算性质和集合的运算性质可得

$$
A=A \cup (B \cap B^c)=(A\cap B)\cup (A\cap B^c)=AB+AB^c
$$

#### Q9

(a) 证明：由于 $B-A=B \cap A^c$，我们有

$$
A+(B-A)=A \cup (B \cap A^c)=(A \cup B) \cap (A \cup A^c)= (A \cup B)\cap \Omega = A\cup B =A+B
$$

由于
$$

A\cap (B-A)=A\cap (B\cap A^c)=(A\cap A^c)\cap B=\emptyset
$$

所以等式右边两事件互斥。

(b) 证明：由于

$$
(A-B)+AB=(A\cap B^c)\cup (A\cap B)=A \cup (B^c \cap B)=A\cup \emptyset = A
$$

结合(a)中的结论可得，

$$
(A+B)=A+(B-A)=((A-B)+AB)+(B-A)=(A-B)+(B-A)+AB
$$

由(a)得 $A \cap (B-A)=\emptyset$，故 $((A-B)\cup AB) \cap (B-A)=\emptyset$，即

$$
((A-B)\cap(B-A))\cup(AB\cap(B-A))=\emptyset
$$

所以 $(A-B)\cap(B-A)=\emptyset$ 且 $AB\cap(B-A)=\emptyset$，即 $B-A$ 与 $A-B$ 互斥，且与 $AB$ 互斥。

又 $(A-B)\cap AB=A \cap B^c \cap A \cap B=\emptyset$，所以 $A-B$ 与 $AB$ 也互斥

综上，等式右边三事件两两互斥。

#### Q10

$(A+B)-(A-B)=(A \cup B)\cap (A \cap B^c)^c=(A \cup B)\cap (A^c\cup B)=(A \cap A^c)\cup B=\emptyset \cup B=B$

#### Q11

构造以下事件 $B_1,B_2,...,B_n$，

$$
B_1=A_1
$$

$$
B_2=(A_1+A_2)-B_1
$$

$$
B_3=(A_1+A_2+A_3)-(B_1+B_2)
$$

$$
......
$$

$$
B_n=\sum_{i=1}^{n}A_i-\sum_{i=1}^{n-1}B_i
$$

显然，对于任意 $1\leq i<j \leq n$，恒有 $B_i \cap B_j=\emptyset$，即 $B_1,B_2,...,B_n$ 两两互斥。

同时，我们有

$$
\sum_{i=1}^{n}B_i=\sum_{i=1}^{n-1}B_i+B_n=\sum_{i=1}^{n-1}B_i+(\sum_{i=1}^{n}A_i-\sum_{i=1}^{n-1}B_i)=\sum_{i=1}^{n}A_i
$$

所以 $A_1,A_2,...,A_n$ 之和能被表示为 $B_1,B_2,...,B_n$ 之和。

#### Q12

Python已安装。
