# 微观到宏观支撑盒的 Jacobian 推导与适用边界

这份笔记讨论一个自然但需要精确定义的想法：若微观尺度上的干预支撑集是一个盒，并且宏观变量由编码器或粗粒化映射给出，那么宏观尺度上的干预支撑盒能否由微观盒边长和编码器 Jacobian 直接推出。

结论先说在前面：概率密度之间的变换关系可以从 normalizing flow 的换元公式或更一般的 coarea 公式严格推出；但宏观盒边长向量不是由密度比公式自动推出的。严格对象是宏观支撑集

$$
\Omega_{\boldsymbol{z}}=\psi(\Omega_{\boldsymbol{x}}),
$$

而最小轴对齐宏观盒的边长必须由每个宏观坐标的取值范围定义。只有在仿射映射、局部近似线性、或各坐标单调且符号稳定等额外条件下，才能把它写成 Jacobian 作用在微观边长向量上的形式。

## 1. 设定：微观盒、编码器和宏观支撑集

设微观状态为

$$
\boldsymbol{x}\in\mathbb{R}^n,
$$

宏观状态为

$$
\boldsymbol{z}=\psi(\boldsymbol{x})\in\mathbb{R}^m,
$$

其中

$$
\psi:\mathbb{R}^n\to\mathbb{R}^m
$$

是一个可微编码器、粗粒化映射或宏观表征映射。微观干预支撑集取为轴对齐盒：

$$
\Omega_{\boldsymbol{x}}
(\boldsymbol{c},\boldsymbol{L}_{\mathrm{micro}})
=
\prod_{i=1}^n
\left[
c_i-\frac{L_{\mathrm{micro},i}}{2},
c_i+\frac{L_{\mathrm{micro},i}}{2}
\right].
$$

这里

$$
\boldsymbol{L}_{\mathrm{micro}}
=
(L_{\mathrm{micro},1},\ldots,L_{\mathrm{micro},n})^\top
$$

是微观盒的边长向量。若沿用现有连续 EI 的有界支撑最大熵口径，则微观参考干预分布为

$$
q_{\boldsymbol{x}}^{\max}(\boldsymbol{x})
=
\frac{1}{|\Omega_{\boldsymbol{x}}|}
\mathbf{1}_{\Omega_{\boldsymbol{x}}}(\boldsymbol{x}),
\qquad
|\Omega_{\boldsymbol{x}}|
=
\prod_{i=1}^n L_{\mathrm{micro},i}.
$$

宏观可达支撑集由微观支撑集推送得到：

$$
\Omega_{\boldsymbol{z}}
=
\psi(\Omega_{\boldsymbol{x}})
=
\{\psi(\boldsymbol{x}):\boldsymbol{x}\in\Omega_{\boldsymbol{x}}\}.
$$

这个定义表达了随附性约束：宏观状态必须有微观实现。它也避免了在宏观隐空间中任意指定一个盒，从而采到没有微观对应物的宏观状态。

## 2. 密度比：严格结果是标量 Jacobian，而不是边长公式

### 2.1 可逆同维情形

先考虑

$$
m=n
$$

且

$$
\psi:\Omega_{\boldsymbol{x}}\to\Omega_{\boldsymbol{z}}
$$

在支撑集上是微分同胚。记编码器 Jacobian 矩阵为

$$
\mathbf{J}_{\psi}(\boldsymbol{x})
=
\frac{\partial\psi(\boldsymbol{x})}{\partial\boldsymbol{x}}
\in\mathbb{R}^{n\times n}.
$$

由 normalizing flow 的换元公式可得

$$
q_{\boldsymbol{z}}(\boldsymbol{z})
=
q_{\boldsymbol{x}}
\!\left(\psi^{-1}(\boldsymbol{z})\right)
\left|
\det
\mathbf{J}_{\psi}
\!\left(\psi^{-1}(\boldsymbol{z})\right)
\right|^{-1}.
$$

等价地，对任意

$$
\boldsymbol{x}\in\Omega_{\boldsymbol{x}},
\qquad
\boldsymbol{z}=\psi(\boldsymbol{x}),
$$

有

$$
\frac{
q_{\boldsymbol{x}}(\boldsymbol{x})
}{
q_{\boldsymbol{z}}(\psi(\boldsymbol{x}))
}
=
\left|
\det \mathbf{J}_{\psi}(\boldsymbol{x})
\right|.
$$

因此，若把“宏微观概率分布之间的比”理解为

$$
q_{\boldsymbol{x}}(\boldsymbol{x})/
q_{\boldsymbol{z}}(\psi(\boldsymbol{x})),
$$

那么严格出现的是标量体积 Jacobian

$$
G(\boldsymbol{x})
=
\left|
\det \mathbf{J}_{\psi}(\boldsymbol{x})
\right|,
$$

而不是完整的矩阵

$$
\mathbf{J}_{\psi}(\boldsymbol{x}).
$$

若反过来写

$$
q_{\boldsymbol{z}}(\psi(\boldsymbol{x}))/
q_{\boldsymbol{x}}(\boldsymbol{x}),
$$

则对应

$$
G(\boldsymbol{x})^{-1}.
$$

这就是 normalizing flow 文献中密度变换项的来源：可逆变换通过 Jacobian 行列式改变体积元素，而不是直接给出每个坐标轴方向的盒边长。

### 2.2 降维情形：coarea Jacobian

宏观建模更常见的是

$$
m<n,
$$

即编码器将高维微观状态压缩为低维宏观状态。此时普通 flow 的一一换元公式不再适用，因为一个宏观状态通常对应一整个微观 fiber：

$$
\psi^{-1}(\boldsymbol{z})
=
\{\boldsymbol{x}\in\Omega_{\boldsymbol{x}}:\psi(\boldsymbol{x})=\boldsymbol{z}\}.
$$

若

$$
\psi
$$

是足够光滑且几乎处处满秩，则 coarea 公式给出合适的替代形式。定义

$$
G_m(\boldsymbol{x})
=
\sqrt{
\det
\left(
\mathbf{J}_{\psi}(\boldsymbol{x})
\mathbf{J}_{\psi}(\boldsymbol{x})^\top
\right)
}.
$$

那么宏观 pushforward 密度满足

$$
q_{\boldsymbol{z}}(\boldsymbol{z})
=
\int_{\Omega_{\boldsymbol{x}}\cap\psi^{-1}(\boldsymbol{z})}
\frac{
q_{\boldsymbol{x}}(\boldsymbol{x})
}{
G_m(\boldsymbol{x})
}
d\mathcal{H}^{n-m}(\boldsymbol{x}),
$$

其中

$$
d\mathcal{H}^{n-m}
$$

是 fiber 上的 Hausdorff 测度。这个公式说明两点：

1. 降维时，宏观密度由整个 fiber 上的微观密度累加得到；
2. Jacobian 进入公式的是标量面积因子

$$
G_m(\boldsymbol{x}),
$$

而不是直接作为宏观盒边长矩阵。

因此，在降维粗粒化中，不能写一个简单的点态比值

$$
q_{\boldsymbol{x}}/q_{\boldsymbol{z}}
$$

来替代全部 fiber 积分。若要给宏观干预以微观语义，还必须同时指定 fiber 内的 lift kernel，即在给定宏观状态时如何选择对应的微观实现。

## 3. 支撑盒边长：严格定义来自坐标范围

宏观支撑集严格定义为

$$
\Omega_{\boldsymbol{z}}
=
\psi(\Omega_{\boldsymbol{x}}).
$$

但它一般不是轴对齐盒。为了继续使用有界盒上的最大熵干预，可以取包含它的最小轴对齐盒：

$$
\mathcal{B}_{\boldsymbol{z}}
=
\prod_{j=1}^m
\left[
\underline{z}_j,\overline{z}_j
\right],
$$

其中

$$
\underline{z}_j
=
\inf_{\boldsymbol{x}\in\Omega_{\boldsymbol{x}}}
\psi_j(\boldsymbol{x}),
\qquad
\overline{z}_j
=
\sup_{\boldsymbol{x}\in\Omega_{\boldsymbol{x}}}
\psi_j(\boldsymbol{x}).
$$

于是宏观盒边长的严格定义是

$$
L_{\mathrm{macro},j}
=
\overline{z}_j-\underline{z}_j
=
\sup_{\boldsymbol{x}\in\Omega_{\boldsymbol{x}}}
\psi_j(\boldsymbol{x})
-
\inf_{\boldsymbol{x}\in\Omega_{\boldsymbol{x}}}
\psi_j(\boldsymbol{x}).
$$

写成向量形式：

$$
\boldsymbol{L}_{\mathrm{macro}}
=
\operatorname{range}_{\Omega_{\boldsymbol{x}}}\psi.
$$

这是最稳妥的定义。它不依赖线性近似，也不要求

$$
\Omega_{\boldsymbol{z}}
$$

本身是盒。

## 4. 仿射情形：精确公式是 \(|\mathbf{A}|\boldsymbol{L}_{\mathrm{micro}}\)

现在考虑最清楚的情形：

$$
\psi(\boldsymbol{x})
=
\mathbf{A}\boldsymbol{x}+\boldsymbol{b},
\qquad
\mathbf{A}\in\mathbb{R}^{m\times n}.
$$

此时

$$
\mathbf{J}_{\psi}(\boldsymbol{x})
\equiv
\mathbf{A}.
$$

微观盒的像

$$
\psi(\Omega_{\boldsymbol{x}})
$$

一般是一个 zonotope，而不是轴对齐盒。第

$$
j
$$

个宏观坐标为

$$
z_j
=
\sum_{i=1}^n A_{ji}x_i+b_j.
$$

在微观盒内，它的最大值在每个

$$
x_i
$$

按

$$
A_{ji}
$$

符号取上端点或下端点时取得；最小值在相反端点取得。因此

$$
L_{\mathrm{macro},j}
=
\sum_{i=1}^n
|A_{ji}|
L_{\mathrm{micro},i}.
$$

即

$$
\boldsymbol{L}_{\mathrm{macro}}
=
|\mathbf{A}|\boldsymbol{L}_{\mathrm{micro}},
$$

其中

$$
|\mathbf{A}|_{ji}=|A_{ji}|.
$$

这一定理说明，若要从矩阵乘法得到轴对齐盒边长，正确矩阵通常不是

$$
\mathbf{A},
$$

而是元素级绝对值矩阵

$$
|\mathbf{A}|.
$$

带符号公式

$$
\boldsymbol{L}_{\mathrm{macro}}
=
\mathbf{A}\boldsymbol{L}_{\mathrm{micro}}
$$

只有在非常特殊的情形才成立，例如每个宏观坐标对所有参与的微观坐标都是非负单调的，或映射本身已经与坐标轴对齐且符号不会造成抵消。

## 5. 非线性情形：平均绝对 Jacobian 是近似，不是定理

若

$$
\psi
$$

是非线性的，则在每个点有局部线性化：

$$
\psi(\boldsymbol{x}+\Delta\boldsymbol{x})
\approx
\psi(\boldsymbol{x})
+
\mathbf{J}_{\psi}(\boldsymbol{x})
\Delta\boldsymbol{x}.
$$

因此，一个自然的工程近似是把仿射公式中的

$$
|\mathbf{A}|
$$

替换为盒内平均的绝对 Jacobian：

$$
\boldsymbol{L}_{\mathrm{macro}}
\approx
\mathbb{E}_{\boldsymbol{x}\sim
\mathrm{Unif}(\Omega_{\boldsymbol{x}})}
\left[
|\mathbf{J}_{\psi}(\boldsymbol{x})|
\right]
\boldsymbol{L}_{\mathrm{micro}}.
$$

等价地，

$$
\boldsymbol{L}_{\mathrm{macro}}
\approx
\frac{1}{|\Omega_{\boldsymbol{x}}|}
\int_{\Omega_{\boldsymbol{x}}}
|\mathbf{J}_{\psi}(\boldsymbol{x})|
d\boldsymbol{x}
\,
\boldsymbol{L}_{\mathrm{micro}}.
$$

这里必须除以

$$
|\Omega_{\boldsymbol{x}}|
$$

得到平均；若直接使用未归一化积分，量纲会多出一个微观体积因子。

这个近似是合理的操作性估计，尤其当编码器在盒内近似仿射、符号稳定、且没有严重折叠时。但它不能作为一般定理。

特别地，在降维情形

$$
\psi:\mathbb{R}^n\to\mathbb{R}^m,
\qquad
m<n,
$$

这个近似仍应写成一个

$$
m\times n
$$

矩阵乘以

$$
n
$$

维微观边长向量：

$$
\widehat{\boldsymbol{L}}_{\mathrm{macro}}
=
\overline{\mathbf{S}}_{\psi}(\Omega_{\boldsymbol{x}})
\boldsymbol{L}_{\mathrm{micro}},
\qquad
\overline{S}_{ji}
=
\mathbb{E}_{\boldsymbol{x}\sim
\mathrm{Unif}(\Omega_{\boldsymbol{x}})}
\left[
\left|
\frac{\partial\psi_j}{\partial x_i}
(\boldsymbol{x})
\right|
\right].
$$

也就是

$$
\widehat{L}_{\mathrm{macro},j}
=
\sum_{i=1}^n
\mathbb{E}_{\boldsymbol{x}\sim
\mathrm{Unif}(\Omega_{\boldsymbol{x}})}
\left[
\left|
\frac{\partial\psi_j}{\partial x_i}
(\boldsymbol{x})
\right|
\right]
L_{\mathrm{micro},i},
\qquad
j=1,\ldots,m.
$$

这里的

$$
\overline{\mathbf{S}}_{\psi}
$$

不是 coarea 公式中的标量

$$
G_m(\boldsymbol{x})
=
\sqrt{\det(\mathbf{J}_{\psi}\mathbf{J}_{\psi}^\top)}.
$$

后者描述的是从微观到宏观的局部

$$
m
$$

维面积因子，并且在 pushforward 密度中还必须与 fiber 积分一起出现；它不能直接给出每个宏观坐标轴方向的盒边长。

一般情况下更稳妥的是范围定义：

$$
L_{\mathrm{macro},j}
=
\sup_{\boldsymbol{x}\in\Omega_{\boldsymbol{x}}}
\psi_j(\boldsymbol{x})
-
\inf_{\boldsymbol{x}\in\Omega_{\boldsymbol{x}}}
\psi_j(\boldsymbol{x}),
$$

或使用上界：

$$
L_{\mathrm{macro},j}
\le
\sum_{i=1}^n
\left(
\sup_{\boldsymbol{x}\in\Omega_{\boldsymbol{x}}}
\left|
\frac{\partial \psi_j}{\partial x_i}
(\boldsymbol{x})
\right|
\right)
L_{\mathrm{micro},i}.
$$

这个上界来自多元均值定理。它比平均 Jacobian 更保守，但作为支撑盒外包络更安全。

## 6. 三个反例与一个精确特例

### 6.1 旋转或剪切：带符号矩阵乘法会抵消边长

令

$$
\mathbf{A}
=
\begin{pmatrix}
1 & -1
\end{pmatrix},
\qquad
\boldsymbol{L}_{\mathrm{micro}}
=
\begin{pmatrix}
1\\
1
\end{pmatrix}.
$$

若使用带符号公式，则

$$
\mathbf{A}\boldsymbol{L}_{\mathrm{micro}}
=
0.
$$

但实际宏观变量为

$$
z=x_1-x_2.
$$

当

$$
x_1,x_2\in[-1/2,1/2],
$$

时，

$$
z\in[-1,1],
\qquad
L_{\mathrm{macro}}=2.
$$

正确的仿射轴对齐盒公式给出

$$
|\mathbf{A}|
\boldsymbol{L}_{\mathrm{micro}}
=
\begin{pmatrix}
1 & 1
\end{pmatrix}
\begin{pmatrix}
1\\
1
\end{pmatrix}
=2.
$$

因此，边长不能用带符号方向相加；边长是范围，应使用绝对贡献或直接求上确界与下确界。

### 6.2 折叠映射 \(z=x^2\)：平均绝对导数会高估支撑宽度

令

$$
x\in[-1,1],
\qquad
z=\psi(x)=x^2.
$$

严格宏观支撑为

$$
z\in[0,1],
\qquad
L_{\mathrm{macro}}=1.
$$

但

$$
|\psi'(x)|=2|x|.
$$

若用一维平均绝对导数近似，则

$$
\mathbb{E}_{x\sim\mathrm{Unif}([-1,1])}
|\psi'(x)|
=
1,
\qquad
L_{\mathrm{micro}}=2,
$$

从而得到

$$
L_{\mathrm{macro}}
\approx
2,
$$

高于真实宽度

$$
1.
$$

原因是

$$
x^2
$$

把正负两侧折叠到同一个宏观区间。局部伸缩的绝对值累积了两侧长度，但支撑范围只看最终覆盖到的区间。

### 6.3 单调一维映射：积分导数给出精确宽度

若

$$
x\in[a,b],
\qquad
z=\psi(x),
$$

且

$$
\psi'(x)\ge 0
$$

在区间内成立，则

$$
L_{\mathrm{macro}}
=
\psi(b)-\psi(a)
=
\int_a^b \psi'(x)\,dx
=
\int_a^b |\psi'(x)|\,dx.
$$

若

$$
\psi'(x)\le 0,
$$

则同样有

$$
L_{\mathrm{macro}}
=
\psi(a)-\psi(b)
=
\int_a^b |\psi'(x)|\,dx.
$$

因此，一维单调情形下，积分绝对导数与支撑宽度一致。这个特例解释了为什么 Jacobian 积分公式在局部单调、无折叠的情形下看起来合理。

### 6.4 多维单调情形：路径积分可精确，体积平均仍未必精确

若每个

$$
\psi_j
$$

对每个坐标

$$
x_i
$$

都保持固定单调方向，则最大值和最小值落在两个相对顶点。此时

$$
L_{\mathrm{macro},j}
=
\psi_j(\boldsymbol{v}_j^+)
-
\psi_j(\boldsymbol{v}_j^-),
$$

其中

$$
\boldsymbol{v}_j^+,\boldsymbol{v}_j^-
$$

按各偏导数的符号选取盒顶点。这个差值可以写成连接两个顶点的路径积分，但一般不能写成整个盒上的体积平均 Jacobian 再乘以

$$
\boldsymbol{L}_{\mathrm{micro}}.
$$

体积平均公式仍然是局部线性近似，而不是多维单调情形的严格定理。

## 7. 对原始设想的判断

原始设想可以拆成两个命题。

第一个命题是：宏微观概率分布之间的比由编码器 Jacobian 决定。这个命题在同维可逆 flow 中是严格成立的，但更准确地说，是由

$$
\left|\det\mathbf{J}_{\psi}\right|
$$

这个标量体积因子决定；在降维粗粒化中则由 coarea Jacobian

$$
G_m(\boldsymbol{x})
=
\sqrt{\det(\mathbf{J}_{\psi}\mathbf{J}_{\psi}^\top)}
$$

和 fiber 积分共同决定。

即使进一步假设同维可逆、宏观像集也恰好可以用一个盒来表示，体积关系也只给出一个标量约束。一般地，

$$
|\Omega_{\boldsymbol{z}}|
=
\int_{\Omega_{\boldsymbol{x}}}
\left|
\det\mathbf{J}_{\psi}(\boldsymbol{x})
\right|
d\boldsymbol{x}.
$$

若

$$
\Omega_{\boldsymbol{z}}
=
\prod_{j=1}^n
[a_j,a_j+L_{\mathrm{macro},j}],
$$

则

$$
\prod_{j=1}^n L_{\mathrm{macro},j}
=
\int_{\Omega_{\boldsymbol{x}}}
\left|
\det\mathbf{J}_{\psi}(\boldsymbol{x})
\right|
d\boldsymbol{x}.
$$

在仿射同维情形，这退化为

$$
\prod_{j=1}^n L_{\mathrm{macro},j}
=
\left|\det\mathbf{A}\right|
\prod_{i=1}^n L_{\mathrm{micro},i}.
$$

这仍然只是边长乘积的约束。满足同一个乘积的边长向量有无穷多个，所以体积守恒或体积缩放不能单独决定

$$
\boldsymbol{L}_{\mathrm{macro}}.
$$

第二个命题是：若各尺度支撑集都取 box，且已知微观边长

$$
\boldsymbol{L}_{\mathrm{micro}},
$$

则宏观边长可以由

$$
\boldsymbol{L}_{\mathrm{micro}}
$$

与盒内积分的

$$
\mathbf{G}
$$

相乘得到。这个命题需要修正：

1. 若

$$
\mathbf{G}
$$

指体积 Jacobian 或 coarea Jacobian，它只能约束体积或密度，不足以确定每个坐标轴方向的盒边长。
2. 若

$$
\mathbf{G}
$$

指编码器 Jacobian 矩阵，则边长公式应使用元素级绝对值，并且一般应使用平均或上界：

$$
\boldsymbol{L}_{\mathrm{macro}}
\approx
\mathbb{E}_{\Omega_{\boldsymbol{x}}}
\left[
|\mathbf{J}_{\psi}(\boldsymbol{x})|
\right]
\boldsymbol{L}_{\mathrm{micro}},
$$

或

$$
\boldsymbol{L}_{\mathrm{macro}}
\le
\mathbf{S}
\boldsymbol{L}_{\mathrm{micro}},
\qquad
S_{ji}
=
\sup_{\boldsymbol{x}\in\Omega_{\boldsymbol{x}}}
\left|
\frac{\partial\psi_j}{\partial x_i}(\boldsymbol{x})
\right|.
$$

3. 严格宏观盒边长仍应定义为坐标范围：

$$
L_{\mathrm{macro},j}
=
\sup_{\Omega_{\boldsymbol{x}}}\psi_j
-
\inf_{\Omega_{\boldsymbol{x}}}\psi_j.
$$

因此，一个更稳妥的写法是：

> 若编码器在微观支撑盒内近似仿射、无显著折叠，并且各宏观坐标对微观坐标的敏感性符号基本稳定，则可用盒内平均绝对 Jacobian 将微观盒边长推到宏观盒边长；但一般情形下，宏观支撑盒必须由编码器像集的坐标范围定义，密度比中的 Jacobian determinant 只能给出体积缩放或 fiber 面积缩放，不能单独决定边长向量。

## 8. 与连续 EI 支撑盒口径的关系

现有连续 EI 口径强调：在连续空间中，最大熵干预必须相对于某个约束来定义。若支撑集可信且有界，则固定支撑集上的均匀分布是自然的最大熵分布；若只固定协方差，则高斯分布才是相应最大熵分布。

微观到宏观支撑盒的推导正好补上了“宏观支撑集从哪里来”这个步骤。推荐流程是：

1. 先由数据覆盖、物理约束和模型可信区域确定

$$
\Omega_{\boldsymbol{x}}.
$$

2. 对每个宏观映射

$$
\psi
$$

定义真实宏观可达支撑

$$
\Omega_{\boldsymbol{z}}
=
\psi(\Omega_{\boldsymbol{x}}).
$$

3. 若计算实现必须使用轴对齐盒，则取

$$
\mathcal{B}_{\boldsymbol{z}}
=
\prod_j
\left[
\inf_{\Omega_{\boldsymbol{x}}}\psi_j,
\sup_{\Omega_{\boldsymbol{x}}}\psi_j
\right].
$$

4. 若为了效率需要近似，则在报告中明确说明使用的是平均绝对 Jacobian 近似或 sup-Jacobian 外包络，而不是由 flow 密度公式严格推出的边长定理。

这样写可以同时保留三个要求：宏观状态随附于微观可达状态；宏观最大熵干预有明确支撑约束；以及不同尺度上的 EI 比较不会把干预盒边界隐藏成任意超参数。

## 参考文献

- Rezende, D. J., & Mohamed, S. (2015). Variational Inference with Normalizing Flows. *Proceedings of Machine Learning Research*, 37, 1530--1538. https://proceedings.mlr.press/v37/rezende15
- Papamakarios, G., Nalisnick, E., Rezende, D. J., Mohamed, S., & Lakshminarayanan, B. (2021). Normalizing Flows for Probabilistic Modeling and Inference. *Journal of Machine Learning Research*, 22(57), 1--64. https://jmlr.org/papers/v22/19-1028.html
- Federer, H. (1969). *Geometric Measure Theory*. Springer. 用于 area formula 与 coarea formula 的标准测度论背景。
- Evans, L. C., & Gariepy, R. F. (2015). *Measure Theory and Fine Properties of Functions* (Revised edition). CRC Press. 可作为 coarea formula 的现代参考。
- 本仓库现有连续 EI 口径见 [研究框架](../研究框架.md) 第 4 节以及 [研究框架_附录](../研究框架_附录.md) 附录 C、D：其中已将连续最大熵干预明确限定为有界支撑盒或协方差约束下的相应最大熵分布。
