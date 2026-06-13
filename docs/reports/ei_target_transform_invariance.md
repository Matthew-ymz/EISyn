# 单源 EI 对目标可逆变换的不变性

## 问题

本实验验证单源有效信息的基本性质：

$$
\operatorname{EI}(S\to Z)=I(S;Z)=I(S;g(Z)),
$$

其中 \(g\) 必须是目标变量上的可逆变换。实验同时比较解析真值、分位数 histogram 和 transport-map 估计，并使用非可逆平方变换作为负对照。

## 解析推导

基础高斯通道为

$$
S\sim\mathcal N(0,1),\qquad
Z=S+\epsilon,\qquad
\epsilon\sim\mathcal N(0,0.5^2).
$$

因此解析 EI 为

$$
I(S;Z)
=\frac12\log_2\left(1+\frac{1}{0.5^2}\right)
=\frac12\log_2 5
=1.160964\ \text{bits}.
$$

若 \(g\) 可逆且可微，则连续熵在变量变换下满足

$$
h(g(Z))=h(Z)+\mathbb E\log_2|g'(Z)|.
$$

条件熵中的 Jacobian 项完全相同：

$$
h(g(Z)\mid S)
=h(Z\mid S)+\mathbb E\log_2|g'(Z)|.
$$

所以

$$
\begin{aligned}
I(S;g(Z))
&=h(g(Z))-h(g(Z)\mid S)\\
&=h(Z)-h(Z\mid S)\\
&=I(S;Z).
\end{aligned}
$$

若 \(g\) 不可逆，该抵消一般不成立。根据数据处理不等式，

$$
I(S;g(Z))\le I(S;Z),
$$

并且当变换丢失与源有关的信息时取严格小于。

## 实验协议

- 每个 seed 采样 `50,000` 组共享的 \(S,\epsilon\)。
- 使用 `12` 个独立 seed。
- 可逆线性扫描：

  $$
  g_c(Z)=cZ,\qquad c\in\{-10,-2,-0.1,0.1,2,10\}.
  $$

- 可逆非线性变换：

  $$
  g_{\mathrm{nl}}(Z)=Z+0.2Z^3,
  $$

  其导数 \(1+0.6Z^2>0\)，因此全局可逆。
- 非可逆负对照：

  $$
  g_{\mathrm{sq}}(Z)=Z^2.
  $$

- histogram 使用 `12` 个分位数箱。
- transport map 使用仓库中的三阶 polynomial triangular transport-map 互信息估计器。

![单源 EI 对目标可逆变换的不变性](../../fig/ei_target_transform_invariance/ei_target_transform_invariance.png)

## 结果

### 可逆线性缩放

| 方法 | 所有非零 \(c\) 下的 EI |
| --- | ---: |
| 解析真值 | `1.160964` bits |
| Quantile histogram | `1.047479 ± 0.005206` bits |
| Transport map | `1.160208 ± 0.004835` bits |

两个数值估计器在所有正负非零缩放系数下都逐点保持不变。transport-map 结果接近解析值，误差约 `-0.000756` bits。histogram 因有限分辨率而低估绝对 EI，但分位数标签只发生保持或反转，因此严格保留缩放不变性。

### 可逆非线性与非可逆负对照

| 目标变换 | 理论 EI | Histogram EI | Transport-map EI |
| --- | ---: | ---: | ---: |
| Identity | `1.160964` | `1.047479 ± 0.005206` | `1.160208 ± 0.004835` |
| \(Z+0.2Z^3\) | `1.160964` | `1.047479 ± 0.005206` | `0.992822 ± 0.020485` |
| \(Z^2\) | 严格小于原 EI | `0.482449 ± 0.005233` | `0.000273 ± 0.000185` |

可逆三次变换不改变目标样本的秩，因此分位数 histogram 仍逐点不变。理论 EI 也严格不变。但是，当前有限阶 transport-map 密度模型对三次变换后的非高斯联合密度存在模型错配，因而低估 EI。这个偏差不是理论不变性失效，而是估计器没有完全恢复真实密度。

平方变换丢失目标符号，因此真实 EI 必然下降。histogram 明确读出下降。当前 transport map 对平方后的多分支、非高斯依赖几乎无法读出，结果接近零；它可以作为估计器失配诊断，但不能被解释为真实 \(I(S;Z^2)\) 接近零。

## 结论

实验支持以下区分：

1. **理论性质**：精确 EI 等于互信息，因此对任意可逆目标变换严格不变。
2. **分位数 histogram**：对严格单调变换天然保持秩标签，因此容易表现出严格不变，但绝对 EI 受分箱分辨率限制。
3. **Transport map**：对模型能够表达的线性缩放准确恢复不变性；对复杂可逆非线性变换可能因密度模型错配而偏离理论值。
4. **非可逆变换**：可能真实丢失 EI，不能应用可逆变换不变性。

因此，观察到估计 EI 随可逆目标变换变化时，应首先检查估计器表达能力与数值误差，而不是立即认为 EI 的理论性质失效。

## 复现

```bash
python scripts/ei_target_transform_invariance.py
```

机器可读结果保存于 `results/ei_target_transform_invariance/ei_target_transform_invariance.json`。

# kuramoto上的实验

为了详细检查平台位置，进一步使用 `23` 个 coupling 点和 `12` 个 seed。网格在 `kappa=0.05` 附近加密到
`[0.04,0.045,0.05,0.055,0.06]`，并把最大 coupling 扩展到 `5.0`。该诊断只比较独立均匀相位干预下的 Oracle PEID 与 MLP+PEID，并分别保存 Syn、joint EI、single-EI sum 和实际耦合信号 RMS。

![Detailed Kuramoto MLP and Oracle PEID sweep](../../fig/classic_network_dynamics_benchmark/kuramoto_peid_detail_sweep.png)

平台不是从 `kappa=0.05` 才开始，而是在任意正 coupling 处立即出现。对 `x` 分支，

$$
\dot x-1=\kappa\sin(w-x).
$$

当 $\kappa>0$ 时，$\dot x-1$ 只是 $\sin(w-x)$ 的非零可逆线性缩放。互信息对目标的可逆变换不变，因此

$$
I(S;\kappa Z)=I(S;Z),\qquad \kappa\neq 0.
$$

joint EI、两个 single EI 以及它们的差 Syn 都继承这个尺度不变性。当前 histogram PEID 使用分位数离散；正比例缩放也不会改变分位数 bin 标签，所以 Oracle histogram PEID 在所有正 $\kappa$ 上逐点保持相同。

密集结果验证了这一点：

- `kappa=0` 时 Oracle 和修正后的 MLP+PEID Syn 都严格为 `0`。
- 对从 `kappa=0.001` 到 `5.0` 的全部正 coupling，Oracle Syn 始终为 `1.058633 ± 0.007261` bits；Oracle joint EI 始终为 `1.077115` bits，single-EI sum 始终为 `0.018482` bits。
- MLP+PEID Syn 在全部正 coupling 上仅在 `1.0634-1.0981` bits 之间波动；MLP joint EI 为 `1.1292-1.1586` bits，single-EI sum 为 `0.0564-0.0751` bits。它与 Oracle 的小偏差来自 MLP 近似误差和有限样本离散，而不是 coupling 强度趋势。
- 与信息平台相反，实际耦合信号 RMS 从 `kappa=0.001` 的 `0.000706` 线性增长到 `kappa=5` 的 `3.529129`，增大约 `5000` 倍。

因此该 Kuramoto PEID 实验回答的是“是否存在不可约的联合相位门控结构”，而不是“门控系数 $\kappa$ 有多大”。如果目标是估计耦合幅值，应同时报告 SHAP interaction、混合导数平方或信号 RMS 等幅值敏感指标。

需要区分“函数几何”和“前置系数”两个问题。$\sin(x+y)$、$\sin(x-y)$ 与 $\sin(xy)$ 都是二源非可分离映射，但它们并不具有相同的协同信息。使用同一组独立均匀源和相同 histogram PEID 协议时，纯映射的 Oracle Syn 分别约为 `1.942`、`1.936` 和 `1.269` bits；差异来自周期折叠、联合映射的多对一结构，以及单独观察一个源时能够保留多少目标信息。

但对于任一固定纯映射 $f(x,y)$，若目标只有

$$
Z_\alpha=c+\alpha f(x,y),\qquad \alpha>0,
$$

则改变 $\alpha$ 只是对目标做可逆缩放，Syn 不应改变。最小 Oracle 对照中，`alpha*sin(x+y)` 与 `alpha*sin(xy)` 从 `alpha=0.001` 到 `5` 的 Syn 都逐点保持不变。

此前 `sin(xy)` alpha 扫描的目标不是纯缩放，而是

$$
z_{t+1}=0.22z_t+\alpha\sin(x_ty_t)+\eta^z_t.
$$

其中 `0.22*z_t` 和噪声没有随 alpha 同步缩放，因此 alpha 会改变协同信号相对记忆项和噪声的强度。固定均匀干预盒的 Oracle 最小对照中，加入 `0.22*z` 后，Syn 从 `alpha=0.1` 的约 `0.043` 增至 `alpha=1` 的约 `0.674`、`alpha=5` 的约 `1.148` bits。也就是说，之前曲线随 alpha 变化的原因是目标中存在未缩放的竞争项，而不是 `sin(xy)` 与 `sin(x+y)` 在尺度不变性上有本质区别。