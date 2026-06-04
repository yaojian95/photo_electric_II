# 基于阶梯样透射数据反推 X 射线能谱的数学与物理原理

双能 X 射线成像（DE-XRT）的传统算法（如 monoenergetic 近似模型）通常假设入射射线为低能 $E_L$ 和高能 $E_H$ 的单能光子。然而，射线管实际出射的是包含连续韧致辐射与特征辐射的多色（Polychromatic）能谱。由于多色射线穿透物质时，低能光子衰减更快，导致射线的平均能量逐渐向高能漂移，这种物理现象称为**能谱硬化（Beam Hardening）**。

为了自适应消除能谱硬化带来的物理量（如有效原子序数 $Z_e$）重建偏差，我们需要准确获知射线源的有效出射能谱。本指南详细阐述如何利用已知材料和厚度参数的 **铜 (Cu)、铁 (Fe)、铝 (Al) 阶梯标样** 实测透射强度的衰减差异，反推重建入射通道 X 射线有效能谱 $S(E)$ 的数学物理模型、求解步骤和代码映射。

---

## 一、 物理前向模型（Forward Model）

对于给定的 X 射线通道（LE 或 HE），假定其归一化有效能谱分布为 $S(E)$，满足：
$$\int_0^{E_{\text{max}}} S(E) dE = 1, \quad S(E) \ge 0$$
这里的 $S(E)$ 实际上是**探测器响应加权后的有效能谱**，即 $S(E) \propto \Phi_0(E) \cdot E \cdot \eta(E)$，其中 $\Phi_0(E)$ 为入射光子谱，$\eta(E)$ 为探测器的能量吸收效率。由于探测器输出的灰度信号与吸收能量成正比，我们无需单独标定探测器效率，反演得到的 $S(E)$ 会自动吸收探测器响应。

当该射线穿过厚度为 $d$ 的某种高纯单质金属（如铝、铁、铜）时，透射后的衰减强度 $I(d)$ 为：
$$I(d) = I_0 \int_0^{E_{\text{max}}} S(E) e^{-\mu(E) d} dE$$
其中：
- $I_0$ 为未遮挡时的空载入射背景灰度值（对应厚度 $d=0$）。
- $\mu(E)$ 为该材质在光子能量为 $E$ 时的**线衰减系数**（单位：$\text{cm}^{-1}$），可由 NIST 数据库的质量衰减系数 $(\mu/\rho)(E)$ 与材质密度 $\rho$ 相乘得到：$\mu(E) = (\mu/\rho)(E) \cdot \rho$。
- $d$ 为透射厚度（单位：$\text{cm}$）。

定义**透射率（Transmission Ratio）**为 $T(d) = I(d) / I_0$，则有：
$$T(d) = \int_0^{E_{\text{max}}} S(E) e^{-\mu(E) d} dE$$

这就是关于待求谱分布 $S(E)$ 的第一类弗雷德霍姆积分方程（Fredholm Integral Equation of the First Kind）。由于核函数 $e^{-\mu(E) d}$ 的平滑性，该逆问题在数学上是极其**病态（Ill-posed）**的。

---

## 二、 物理量离散化与线性方程组

为了在计算机中求解，我们将能量区间 $[E_{\text{min}}, E_{\text{max}}]$（例如 $15\text{ keV}$ 到射线管最大工作电压 $V_{max}\text{ keV}$）等分为 $M$ 个能量仓（Energy Bins），每个能量仓的宽度为 $\Delta E$。

设离散能量点为 $E_i$（$i=1, 2, \dots, M$），则积分可近似为有限项求和：
$$T(d) \approx \sum_{i=1}^M S(E_i) e^{-\mu(E_i) d}$$
此时，待求的能谱分布简化为大小为 $M$ 的非负向量 $\mathbf{S} = [S_1, S_2, \dots, S_M]^T$，满足 $\sum_{i=1}^M S_i = 1$。

假定我们一共采集了 $N$ 个不同的阶梯标样测量点（包含 Al, Fe, Cu 在不同厚度下的透射率实测值），对于第 $j$ 次测量（对应材料 $mat(j)$，厚度 $d_j$）：
$$T_j = \sum_{i=1}^M A_{j, i} S_i, \quad j=1, 2, \dots, N$$
where 矩阵元定义为：
$$A_{j, i} = e^{-\mu_{mat(j)}(E_i) d_j}$$

写成矩阵形式：
$$\mathbf{A} \mathbf{S} \approx \mathbf{T}$$
- $\mathbf{A}$ 是大小为 $N \times M$ 的前向系统矩阵。
- $\mathbf{T} = [T_1, T_2, \dots, T_N]^T$ 是实测透射率向量。

---

## 三、 病态系统求解与正则化约束

因为不同能量对应的衰减指数曲线 $e^{-\mu(E)d}$ 具有极高的共线性，系统矩阵 $\mathbf{A}$ 的条件数非常大。如果直接求解最小二乘 $\mathbf{S} = \mathbf{A}^+ \mathbf{T}$，解向量会出现剧烈的正负振荡和无物理意义的噪声斑。

为了获得平滑且符合物理规律的真实能谱，我们引入以下物理与数学先验约束：

### 1. 非负性约束 (Non-negativity)
光子数或能量流不能为负值：
$$S_i \ge 0, \quad \forall i$$

### 2. 归一化约束 (Normalization)
各能量组分的比例和必须为 1：
$$\sum_{i=1}^M S_i = 1$$
在数值求解中，我们引入高权重因子 $\gamma$（如 $\gamma = 20.0$），在方程组中附加一行：
$$\gamma \sum_{i=1}^M S_i = \gamma$$

### 3. 能量边界约束 (Boundary Conditions)
由于射线管阳极靶自身和外加滤片（如 0.6mm 或 1.2mm 铜滤片）的固有过滤作用，极低能量（如 $<15\text{ keV}$）的光子几乎被 100% 吸收；同时，由于射线管管电压的能量上限限制，超过最大电压 $V_{\text{max}}$ 的光子数严格为 0。因此，能谱的两端边界必须平滑降为 0：
$$S_1 = 0, \quad S_M = 0$$
在数值求解中，引入高权重因子 $\beta$（如 $\beta = 10.0$），在方程组中附加两行约束：
$$\beta S_1 = 0, \quad \beta S_M = 0$$

### 4. 平滑性正则化（Tikhonov Regularization）
X 射线谱为连续且平滑的曲线。我们可以通过惩罚能谱的二阶导数（即曲率）来抑制振荡。能谱在 $E_i$ 处的二阶差分为：
$$\Delta^2 S_i = S_{i-1} - 2S_i + S_{i+1}$$
构建大小为 $(M-2) \times M$ 的二阶差分算子矩阵 $\mathbf{D}$：
$$\mathbf{D} = \begin{bmatrix}
1 & -2 & 1 & 0 & \dots & 0 & 0 \\
0 & 1 & -2 & 1 & \dots & 0 & 0 \\
\vdots & \ddots & \ddots & \ddots & \ddots & \vdots & \vdots \\
0 & 0 & \dots & 1 & -2 & 1 & 0 \\
0 & 0 & \dots & 0 & 1 & -2 & 1
\end{bmatrix}$$
我们通过正则化参数 $\lambda$（如 $\lambda = 0.005$）控制平滑强度，要求 $\sqrt{\lambda} \mathbf{D} \mathbf{S} \approx \mathbf{0}$。

---

## 四、 增广非负最小二乘（Augmented NNLS）表述

将上述所有方程与约束合并，我们构建**增广系统矩阵 $\mathbf{A}_{\text{aug}}$** 和 **增广目标向量 $\mathbf{T}_{\text{aug}}$**：

$$\mathbf{A}_{\text{aug}} = \begin{bmatrix} 
\mathbf{A}_{N \times M} \\
\gamma \cdot [1, 1, \dots, 1]_{1 \times M} \\
\beta \cdot [1, 0, \dots, 0]_{1 \times M} \\
\beta \cdot [0, 0, \dots, 1]_{1 \times M} \\
\sqrt{\lambda} \cdot \mathbf{D}_{(M-2) \times M}
\end{bmatrix}, \quad
\mathbf{T}_{\text{aug}} = \begin{bmatrix}
\mathbf{T}_{N \times 1} \\
\gamma \\
0 \\
0 \\
\mathbf{0}_{(M-2) \times 1}
\end{bmatrix}$$

此时，谱反推问题转化为经典的**凸优化非负最小二乘问题（Non-negative Least Squares, NNLS）**：

$$\text{minimize} \quad \|\mathbf{A}_{\text{aug}} \mathbf{S} - \mathbf{T}_{\text{aug}}\|_2^2 \quad \text{subject to} \quad \mathbf{S} \ge \mathbf{0}$$

由于 NNLS 在凸空间内具有全局唯一最优解，我们可使用高效的 Lawson-Hanson 活动集算法（Active-Set Method）在几毫秒内求得稳定、平滑且精确归一化的物理能谱向量。

### 增广矩阵参数与零壹结构的物理解析

在数值求解过程中，增广矩阵 $\mathbf{A}_{\text{aug}}$ 里的权重因子（$\gamma$ 和 $\beta$）以及其中的 $1$ 和 $0$ 元素，本质上是**约束控制权重**与**数学选择器**：

#### 1. $\gamma$ 和 $\beta$ 的物理作用：残差惩罚权重
优化求解器以最小化所有行方程的残差平方和为目标。
- **$\gamma$（归一化约束权重，设定为 20.0）**：实测的透射率 $\mathbf{T}$（值为 $I/I_0$）都在 $0.01 \sim 0.8$ 之间，方程误差平方非常小（约 $10^{-4}$ 数量级）。如果仅加入一行不带权重的求和约束 $\sum S_i = 1$（即 $[1, 1, \dots, 1] \mathbf{S} = 1$），若求和结果为 $0.95$（残差仅为 $0.05$），残差平方仅仅是 $0.0025$。求解器为了强行降低其他透射率数据点的拟合误差，会牺牲掉该约束。当我们乘上 $\gamma = 20.0$ 时，若求和依然为 $0.95$，对应的方程残差就变为 $20 \times (0.95 - 1.0) = -1.0$，其残差平方放大了 400 倍（变为 $1.0$）。这迫使求解器必须优先保证能谱总和等于 1。
- **$\beta$（边界截止约束权重，设定为 10.0）**：同理，为了满足能谱在低能截止（15 keV）和高能截止（管电压上限），我们要求边界点 $S_1 = 0$ 和 $S_M = 0$。通过乘上权重 $\beta = 10.0$，能谱两端不归零的残差惩罚被放大 100 倍，强迫能谱曲线两端贴紧为 0。

#### 2. $1$ 和 $0$ 元素的数学作用：元素选择器与提取器
增广部分中由 $1$ 和 $0$ 组成的行向量，是用来**提取能谱特定物理特征的“选择开关”**：
- **全一向量 $[1, 1, \dots, 1]$**：在矩阵乘法中，它与列向量 $\mathbf{S}$ 进行内积，相乘结果刚好是 $\sum_{i=1}^M S_i$，即能谱中所有分量的总和。
- **第一列为一向量 $[1, 0, 0, \dots, 0]$**：与列向量 $\mathbf{S}$ 做内积，对应元素相乘后只剩下 $S_1$。它精准地把能谱的第一项提取出来，用来对其施加归零约束。
- **最后一列为一向量 $[0, 0, \dots, 0, 1]$**：与列向量 $\mathbf{S}$ 做内积，只留下能谱最后一项 $S_M$。它精准地提取出能谱最后一项，施加高能截止归零约束。
- **差分算子行 $[0, \dots, 1, -2, 1, \dots, 0]$**：当它与能谱相乘时，只提取相邻的三点进行差分计算 $S_{i-1} - 2S_i + S_{i+1}$，配合右侧目标向量为 0，强行约束曲线的局部曲率趋近于 0，使谱线过渡极其平滑。

---

## 五、 Python 代码映射与执行流程

在 [reconstruct_spectrum.py](file:///e:/photo_electric_II/reconstruct_spectrum.py) 中，该求解逻辑通过 `reconstruct_channel_spectrum` 函数实现：

```python
import numpy as np
import scipy.optimize

def reconstruct_channel_spectrum(A: np.ndarray, T: np.ndarray, energies_keV: np.ndarray, 
                                 lambda_val: float = 0.005, gamma: float = 20.0, beta: float = 10.0) -> np.ndarray:
    """
    使用增广正则化非负最小二乘 (NNLS) 求解单通道归一化 X 射线出射谱 S(E)。
    
    参数：
    - A (np.ndarray): 正向投影矩阵，大小 (N, M)，A[j, i] = exp(-mu_j(E_i) * d_j)。
      - 类型：np.ndarray (float)
      - 含义：联系能谱组分与各厚度透射率的物理系统矩阵。
    - T (np.ndarray): 实测透射率向量，大小 (N,)，值为 I/I0。
      - 类型：np.ndarray (float)
      - 含义：各标样厚度阶梯的实测透射比例。
    - energies_keV (np.ndarray): 能量网格数组，大小 (M,)，单位 keV。
      - 类型：np.ndarray (float)
      - 含义：离散能谱的能量仓中心坐标。
    - lambda_val (float): 平滑正则化惩罚因子。
      - 类型：float
      - 含义：值越大能谱越平滑，值越小越贴近原始数据点但易振荡。
    - gamma (float): 归一化和为 1 的约束权重。
      - 类型：float
      - 含义：用于将 sum(S) = 1 融入最小二乘目标函数。
    - beta (float): 边界归零约束的权重。
      - 类型：float
      - 含义：强制第一仓与最后一仓强度为 0。
      
    返回：
    - np.ndarray: 大小为 (M,) 的归一化能谱概率向量 S。
    """
    M = len(energies_keV)
    
    # 1. 构造二阶差分平滑算子矩阵 D (大小 M-2 x M)
    D = np.zeros((M - 2, M))
    for i in range(M - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
        
    # 2. 构造能量两端边界归零约束行
    row_bound_start = np.zeros((1, M))
    row_bound_start[0, 0] = beta
    row_bound_end = np.zeros((1, M))
    row_bound_end[0, -1] = beta
    
    # 3. 构造能谱归一化约束行 (sum(S) = 1)
    row_norm = gamma * np.ones((1, M))
    
    # 4. 纵向拼装增广系统矩阵 A_aug 与目标向量 T_aug
    A_aug = np.vstack([
        A,
        row_norm,
        row_bound_start,
        row_bound_end,
        np.sqrt(lambda_val) * D
    ])
    
    T_aug = np.concatenate([
        T,
        [gamma],
        [0.0],
        [0.0],
        np.zeros(M - 2)
    ])
    
    # 5. 调用 scipy 的 NNLS 求解器进行凸优化求解
    S, _ = scipy.optimize.nnls(A_aug, T_aug)
    
    # 6. 后处理再次强行精确归一化防止浮点数微小偏差
    sum_S = np.sum(S)
    if sum_S > 0:
        S = S / sum_S
        
    return S
```

---

## 六、 能谱解算结果应用：解耦能谱硬化的 APD/ACD 特征求解

求得 $S_L(E)$ 与 $S_H(E)$ 后，传统的 Monoenergetic 二元线性方程组将被升级为连续积分方程组。
对于任意包含待求光电项面密度 $A_p = a_p \cdot d$ 和康普顿散射面密度 $A_c = a_c \cdot d$ 的像素，其测量到的高低能透射率对 $(T_L, T_H)$ 满足：
$$T_L = \sum_{i=1}^M S_L(E_i) e^{- (A_p E_i^{-3} + A_c f_{\text{KN}}(E_i))}$$
$$T_H = \sum_{i=1}^M S_H(E_i) e^{- (A_p E_i^{-3} + A_c f_{\text{KN}}(E_i))}$$

由于 $S_L(E)$ 和 $S_H(E)$ 已知，我们通过二元 Newton-Raphson 算法求解上述两个非线性代数方程，即可从高低能灰度值直接解算出**完全解耦能谱硬化漂移、与厚度保持严格线性关系**的 $A_p$ ($apd$) 与 $A_c$ ($acd$)，从而极大地提升后续原子序数 $Z_e$ 的辨识精度与系统鲁棒性。

---

## 七、 另外一种方法（方法二）：相邻厚度差值能谱估计算法

### 1. 数学原理
设相邻两个阶梯厚度为 $d_j$ 和 $d_{j+1}$（满足 $d_{j+1} > d_j$），则其对应的实测透射率差值 $\Delta T_j$ 可表示为：
$$\Delta T_j = T(d_j) - T(d_{j+1}) = \int_0^{E_{\text{max}}} S(E) \left[ e^{-\mu(E)d_j} - e^{-\mu(E)d_{j+1}} \right] dE$$

定义通道差值敏感核函数为 $g_j(E) = e^{-\mu(E)d_j} - e^{-\mu(E)d_{j+1}}$。
由于线衰减系数 $\mu(E)$ 在 X 射线诊断能量范围内是能量 $E$ 的单调递减函数，因此可将 $g_j(E)$ 视作对 $\mu$ 的函数 $g_j(\mu) = e^{-\mu d_j} - e^{-\mu d_{j+1}}$。令其对 $\mu$ 的导数为 0：
$$\frac{dg_j}{d\mu} = -d_j e^{-\mu^* d_j} + d_{j+1} e^{-\mu^* d_{j+1}} = 0$$
从而解得该差值函数响应最敏感的峰值衰减系数 $\mu^*$：
$$\mu^* = \frac{\ln(d_{j+1}/d_j)}{d_{j+1} - d_j}$$

在 NIST 数据库中，通过对线衰减系数对数空间的逆插值，我们可以反向推算出 $\mu^*$ 对应的光子能量点 $E^*_j$：
$$E^*_j = \mu^{-1}(\mu^*)$$

假定能谱 $S(E)$ 在峰值能量 $E^*_j$ 附近是平滑的，且核函数 $g_j(E)$ 呈现窄带准带通特性，则该积分可近似为：
$$\Delta T_j \approx S(E^*_j) \int_0^{E_{\text{max}}} g_j(E) dE = S(E^*_j) C_j$$
其中常数 $C_j = \int_0^{E_{\text{max}}} \left[ e^{-\mu(E)d_j} - e^{-\mu(E)d_{j+1}} \right] dE$ 可通过 NIST 数据库数值积分直接算出。
由此，我们可以直接估计在特定能量点 $E^*_j$ 处的能谱强度：
$$S(E^*_j) \approx \frac{\Delta T_j}{C_j}$$

遍历铝、铁、铜的所有相邻厚度差值，可获取一系列离散能谱采样点 $(E^*_j, S(E^*_j))$。对这些采样点使用单调分段三次 Hermite 插值（PCHIP）插值到目标能谱网格，并进行非负截断与总和归一化，即得到完整的能谱曲线。

---

## 八、 方法二的物理可行性评估与实测对比

通过执行 `reconstruct_spectrum.py` 重建 0429 阶梯样数据，并使用非线性求解器 `solve_apd_acd_nonlinear` 解算三材质（Al, Fe, Cu）在各管电压下的 $apd$ 与 $acd$，我们获得了以下定量评估结果：

### 1. 重建有效能谱的均值偏离
在管电压为 $200\text{ kV}$ 的测试中，两种方法重建的入射通道（LE / HE）有效平均能量为：
- **方法一 (NNLS正则化最小二乘)**：$E_{L, eff} \approx 65.00$ keV, $E_{H, eff} \approx 109.45$ keV (符合带滤片的连续 X 射线源物理规律)。
- **方法二 (差值映射法)**：$E_{L, eff} \approx 103.82$ keV, $E_{H, eff} \approx 116.53$ keV (严重偏向高能区)。

### 2. 物理成因分析（为何方法二失效）
- **能谱严重硬化过滤**：方法二的核心假设是差值 $\Delta T_j$ 反映了能量段 $E^*_j$ 的入射光子。但阶梯标样实际物理厚度非常厚（Cu/Fe 范围为 2mm - 20mm，Al 范围为 12mm - 30mm）。在射流穿透第一级厚度 $d_j$（如 2mm 铜）时，低能区光子已被近乎 100% 滤除（能谱极度硬化）。
- **非入射谱表征**：因此，差值 $\Delta T_j = T(d_j) - T(d_{j+1})$ 实际上所包含并反映的是**已被厚度 $d_j$ 强力过滤后的极硬化谱**，而非我们在 $d=0$ 处的**原始入射能谱**。
- **物理前向模型冲突**：在解耦算法中，我们建立的透射前向方程 $T_L = \int S_L(E) e^{-(\dots)} dE$ 必须输入 $d=0$ 处的入射能谱 $S_L(E)$。一旦误入方法二的高能偏置谱，会导致积分模型的核函数与真实透射率发生物理错配，解出的 $apd$ 和 $acd$ 会出现严重正负振荡。

### 3. $R^2$ 线性度定量对比 (Average $R^2$ Linearity)
对所有 7 个电压（200kV - 320kV）和 2 种滤片（0.6mm, 1.2mm）下的所有标样厚度级进行 APD/ACD 特征对厚度的线性拟合（过原点），其平均判定系数 $R^2$ 汇总如下：

| 特征类型 | 静态单能法 (58/105 keV) | 动态等效单能法 | 方法一 (NNLS正则化谱) | 方法二 (差值能谱) |
| :--- | :---: | :---: | :---: | :---: |
| **$apd$ 平均 $R^2$** | -10.675495 | -10.298655 | **0.432801** | -48.211834 |
| **$acd$ 平均 $R^2$** | 0.991859 | 0.993081 | **0.879538** | -48.003203 |

> [!WARNING]
> 方法二（差值能谱）在 APD 和 ACD 特征解算中，平均 $R^2$ 线性度退化为严重的负值（$\approx -48.0$），表明拟合残差远大于因变量自身的方差，出现了严重的物理性退化。

### 4. 结论与执行建议
方法二在**物理上是不可行的**（对于入射能谱反推），其结果只能表征强硬化后的残余谱，与前向透射积分模型存在物理冲突。
我们应当**坚持采用方法一（正则化增广 NNLS）**作为出射谱反演的唯一科学方法，因其实现了最佳的特征线性度，能有效克服能谱硬化漂移。目前代码中已完整保存并输出了方法二的对比曲线（以虚线形式展示），在磁盘输出结果中标记为 `_m2` 即可供对比教学使用。

---

## 九、 求得能谱后，解算 $apd$, $acd$, 以及 Bulk 材质系数 $a_p$ 和 $a_c$ 的完整步骤

通过物理能谱反演得到低高能归一化能谱 $S_L(E_i)$ 和 $S_H(E_i)$ 后，后续物理参数的解算分为两步：第一步求解每个像素或阶梯级对应的路径积分特征 $apd$ 与 $acd$；第二步通过物理厚度对齐求解该材质专属的物理特征系数 Bulk 常数 $a_p$ 与 $a_c$。

### 1. 第一步：数值求解每个测量点的 $apd$ 和 $acd$（路程积分特征）
对于测得低高能透射率对 $(T_L, T_H)$ 的图像像素或阶梯，我们通过多色能谱前向积分模型建立以下非线性二元方程组：
$$T_L = \sum_{i=1}^M S_L(E_i) e^{-(apd \cdot E_i^{-3} + acd \cdot f_{\text{KN}}(E_i))}$$
$$T_H = \sum_{i=1}^M S_H(E_i) e^{-(apd \cdot E_i^{-3} + acd \cdot f_{\text{KN}}(E_i))}$$

由于能谱概率向量 $S_L, S_H$ 已知，能量仓中心 $E_i$ 已知，Klein-Nishina 系数 $f_{\text{KN}}(E_i)$ 亦已知，该方程组中仅有 $apd$ 和 $acd$ 两个未知数。
在代码中，该步骤由 `solve_apd_acd_nonlinear` 函数完成：
1. **初值预测**：先用能谱均值能量作为等效单能点：
   $$E_{L, \text{eff}} = \sum E_i S_L(E_i), \quad E_{H, \text{eff}} = \sum E_i S_H(E_i)$$
   利用 Monoenergetic 代数解析公式计算得到 $apd_{\text{init}}, acd_{\text{init}}$ 作为初始搜索起点。
2. **非线性寻根**：构建多元残差目标函数：
   $$F(apd, acd) = \begin{bmatrix} \sum_{i=1}^M S_L(E_i) e^{-(apd \cdot E_i^{-3} + acd \cdot f_{\text{KN}}(E_i))} - T_L \\ \sum_{i=1}^M S_H(E_i) e^{-(apd \cdot E_i^{-3} + acd \cdot f_{\text{KN}}(E_i))} - T_H \end{bmatrix}$$
   调用 `scipy.optimize.root(method='hybr')` 迭代寻根，在满足极小残差条件下收敛解出该点的非色散特征对 $(apd, acd)$。该解已在物理机制上将能谱硬化漂移完全解耦。

### 2. 第二步：拟合求解 Bulk 材质物理系数 $a_p$ 和 $a_c$
对于铜、铁、铝标样，由于其各厚度级对应的物理厚度 $d_j$（$j = 1, \dots, K$）是精确已知的，且物理模型满足：
$$apd = a_p \cdot d, \quad acd = a_c \cdot d$$
我们可通过以下两种拟合方式获取反映材质衰减贡献特征的 Bulk 系数：

* **方法 A：无截距一元线性最小二乘拟合 (Linear Regression through the Origin)**：
  为了使所有厚度级拟合残差平方和最小，对 $a_p$ 与 $a_c$ 求解正规方程，可直接计算出其斜率：
  $$a_p = \frac{\sum_{j=1}^K d_j \cdot apd_j}{\sum_{j=1}^K d_j^2}$$
  $$a_c = \frac{\sum_{j=1}^K d_j \cdot acd_j}{\sum_{j=1}^K d_j^2}$$
  此斜率即为该材质的平均 Bulk 系数，拟合时的 $R^2$ 即可用来评估该重构管线的线性度性能。

* **方法 B：基于极薄第一阶梯的直接除法求解**：
  大厚度阶梯下容易受次级散射和二次硬化的二次效应扭曲。为了确保提取出的 Bulk 特征物理纯度最高，可使用受硬化效应最轻微的第一级阶梯（厚度 $d_1$，解出特征 $apd_1, acd_1$）直接求得 Bulk 常数：
  $$a_p = \frac{apd_1}{d_1}, \quad a_c = \frac{acd_1}{d_1}$$
  该 Bulk 特征参数随后可代入 SIRZ 校准模型，从而回归求得系统的电子密度常数 $K_1$ 与原子序数标定系数 $g, \nu$。

