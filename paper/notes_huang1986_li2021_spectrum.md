# 📄 论文阅读报告：X 射线能谱重建方法

**报告日期：** 2026-06-10  
**阅读论文：**
1. [huang1986.pdf](file:///e:/photo_electric_II/paper/huang1986.pdf) — Huang, Chen & Kase, *Med. Phys.* 13, 707 (1986)  
2. [fphy-09-678171.pdf](file:///e:/photo_electric_II/paper/fphy-09-678171.pdf) — Li, Fan, Cong & Wang, *Front. Phys.* 9, 678171 (2021)

---

## 一、论文一：Huang et al. (1986)

### 1.1 基本信息

| 项目 | 内容 |
|---|---|
| 题目 | Reconstruction of Diagnostic X-Ray Spectra by Numerical Analysis of Transmission Data |
| 作者 | Pin-Hua Huang, Tao-Seng Chen, Kenneth R. Kase |
| 期刊 | *Medical Physics*, Vol. 13, No. 5, pp. 707–710 |
| DOI | 10.1118/1.595834 |
| 年份 | 1986 |

### 1.2 研究背景与动机

在 45～100 kVp 诊断 X 射线能量范围内，直接用能量分辨探测器测量能谱代价高昂且技术复杂（特别是高通量源下难以避免探测器死时间效应）。Huang 等人提出用**窄束透射衰减数据**来**间接反推**X 射线能谱，这是典型的第一类 Fredholm 积分方程逆问题。

### 1.3 核心数学框架

**前向物理模型（First-Kind Fredholm 积分方程）：**

$$T(x) = \int_0^{E_{\max}} S(E) \, e^{-\mu(E) \, x} \, dE$$

其中：
- $T(x)$：通过厚度为 $x$ 的铝滤片后的测量透射率 $I(x)/I_0$  
- $S(E)$：待求的归一化能谱分布（实为探测器响应加权的有效谱）  
- $\mu(E)$：铝在能量 $E$ 处的线衰减系数（来自 NIST 数据）  
- $E_{\max}$：与管电压对应的最大光子能量（kVp）

**离散化为线性方程组：**

将能量范围 $[0, E_{\max}]$ 划分为 $N$ 个等宽能量仓（energy bins），则：

$$T(x_i) \approx \sum_{j=1}^{N} S_j \cdot e^{-\mu_j x_i}, \quad i = 1, \ldots, M$$

矩阵形式：$\mathbf{A} \mathbf{S} \approx \mathbf{T}$，其中 $A_{ij} = e^{-\mu_j x_i}$。

### 1.4 核心算法：迭代数值分析（Iterative Unfolding）

该逆问题是**病态（ill-posed）的**，因为系统矩阵 $\mathbf{A}$ 的条件数极大，直接求逆会导致解振荡。Huang 等人采用迭代展开法（类似 Expectation-Maximization 前身）：

**算法步骤：**
1. **初始化**：给定一个初始正的猜测谱 $S^{(0)}(E)$（例如平坦分布）
2. **正向预测**：用当前谱 $S^{(k)}$ 计算预测透射率 $\hat{T}^{(k)}(x_i)$
3. **误差反馈**：计算实测透射率 $T(x_i)$ 与预测值的比值（乘性更新）
4. **迭代更新**：用加权比值校正每个能量仓的谱强度
5. **收敛判断**：当计算谱引起的透射率与实测数据之差小于阈值，停止迭代

**关键辅助工具：**  
利用 $T'(0)$（透射曲线在零厚度处的斜率）估计初始等效能量，辅助收敛。

### 1.5 实验验证与结论

- **测试范围**：45 ～ 100 kVp 诊断能量区间
- **核对基准**：用已知理论谱生成模拟透射曲线，再从透射数据重建谱
- **关键结果**：重建谱与理论谱吻合良好，**能准确还原钨靶特征 X 射线峰（K 特征线）**
- **意义**：证明铝滤片透射衰减数据是实用有效的能谱间接推算手段

### 1.6 方法的局限性（后续文献补充）

- 该论文仅使用**铝单材质**，线性等厚阶梯排布 → 系统矩阵高度病态（Vandermonde-like structure）
- 对噪声较敏感，需要较多阶梯测量点以稳定求解
- 本质上属于早期的迭代正则化思路，现代文献（Li 2021 等）在此基础上做了大幅改进

---

## 二、论文二：Li, Fan, Cong & Wang (2021)

### 2.1 基本信息

| 项目 | 内容 |
|---|---|
| 题目 | EM Estimation of the X-Ray Spectrum With a Genetically Optimized Step-Wedge Phantom |
| 作者 | Mengzhou Li, Feng-Lei Fan, Wenxiang Cong, Ge Wang |
| 期刊 | *Frontiers in Physics*, Vol. 9, Article 678171 |
| DOI | 10.3389/fphy.2021.678171 |
| 年份 | 2021 |

### 2.2 研究背景与动机

在 CT 等成像系统中，知晓 X 射线管的能谱对于：剂量估算、双能材料分解、伪影校正、探测器性能评估等都至关重要。然而，已有文献（包括 Huang 1986）均面临**逆问题严重病态**的挑战。Li 等人聚焦于通过优化**测量方案**（phantom 设计）本身来降低病态性，而非仅依赖正则化技巧。

### 2.3 核心数学框架

**前向测量模型（与 Huang 1986 一致）：**

$$p_i = \int_E W(E) \exp\left(-l_i \mu_i(E)\right) dE$$

其中 $W(E) = S(E)D(E) / \int S(E)D(E)dE$ 为包含探测器响应的归一化有效谱。

**离散化后的矩阵方程：**

$$p = Aw, \quad A \in \mathbb{R}^{M \times N}, \quad a_{ij} = e^{-l_i \mu_{ij}}$$

### 2.4 核心理论贡献一：病态性分析

**关键发现（等厚线性排布 → Vandermonde 矩阵问题）：**

对于单材质、等间距厚度排列，系统矩阵可分解为：

$$A = VD$$

其中 $V$ 是 Vandermonde 矩阵，$D$ 是对角矩阵。由于 Vandermonde 矩阵的条件数至少以 $O(2^N)$ 的指数速度增长：

$$\text{cond}(A) \geq \frac{d_{\min}}{d_{\max}} \cdot \text{cond}(V) \sim O(2^N)$$

这从**理论**上解释了为何传统线性阶梯排布随测量点数增加，病态性急剧恶化。

### 2.5 核心算法一：遗传算法（GA）优化阶梯厚度

**优化目标：** 在给定材质和测量点数 $M$ 下，最小化系统矩阵的条件数：

$$\min_{\mathbf{l}} \quad \text{cond}_2(A) = \frac{\sigma_{\max}}{\sigma_{\min}}$$

**遗传算法编码：**
- **染色体**：$M$ 个厚度 $\mathbf{l} = [l_1, l_2, \ldots, l_M]$
- **适应度函数**：$-\ln(\text{cond}(A))$（取负对数，越大越好）
- **交叉操作**：随机选择基因位置，按混合比 $r_m$ 交叉父代基因
- **变异操作**：按公式 $l_i \leftarrow l_i - (l_i - l_0) r_s (1 - \gamma)^2$ 调整（非均匀变异，随代数收紧）
- **超参数**：最大代数 500，种群大小 1000，交叉概率 0.65，变异概率 0.1

### 2.6 核心算法二：期望最大化（EM）能谱重建

**EM 乘性更新公式（自动保证非负性）：**

$$w_j^{(k+1)} = \frac{w_j^{(k)}}{\sum_i a_{ij}} \sum_i \frac{a_{ij} p_i}{\sum_{j'} a_{ij'} w_{j'}^{(k)}}$$

**初始化策略（关键！）：**  
由于 EM 算法难以自行恢复高频细节（如钨靶特征 K 线峰），作者将特征线峰的**位置信息**作为先验，初始化时在特征能量处设置更高的初始值，其余位置设为 1。

### 2.7 实验结论（量化对比）

**材质选择建议（N = 22 能量仓，M = 9 测量点）：**

| 材质 | 最优厚度均值序列（cm） | 条件数均值 | 相对稳定性 |
|---|---|---|---|
| **Cu** | 0.002, 0.004, 0.009, 0.021, 0.045, 0.096, 0.201, 0.420, 0.991 | **1994** | ★★★ |
| **Fe** | 0.001, 0.005, 0.013, 0.030, 0.064, 0.135, 0.282, 0.584, 1.367 | **2160** | ★★★ |
| Al | 0.022, 0.090, 0.228, 0.475, 0.892, 1.562, 2.588, 4.051, 4.998 | 77,778 | ★★ |
| PMMA | 0.105, 0.445, 0.989... | 9.47×10¹² | ★（最差） |

**关键结论：**

1. **非线性排布优于线性排布**：条件数降低约 2 个数量级
2. **最优厚度序列呈指数分布**（可用 $y = a \cdot e^{bx}$ 拟合，$R^2 \approx 0.999$）
3. **Cu 和 Fe 在多测量点时优于 Al**；Al 仅在 $M \leq 7$ 时有优势
4. **PMMA 不适合作为阶梯材质**（条件数最差）
5. **15 个优化厚度 > 50 个线性厚度**的重建质量
6. **多材质组合**可进一步降低条件数（Al+Cu+Fe 组合达到 38.6）

---

## 三、两篇论文的核心对比

| 对比维度 | Huang 1986 | Li 2021 |
|---|---|---|
| **核心贡献** | 首次提出透射反推能谱的可行性 | 揭示病态性根源，提出优化阶梯设计 |
| **求解算法** | 迭代数值展开法（EM 前身） | EM 算法 + 初始化先验 |
| **正则化** | 无显式正则化（依赖迭代截止） | 无显式正则化（依赖测量设计优化） |
| **测量材质** | 铝（单材质） | Al, PMMA, Cu, Fe（四材质对比） |
| **厚度排布** | 线性等间距（传统做法） | 遗传算法优化的指数分布 |
| **测量点数** | 未明确规定（约数十个） | 建议 5～15 个即可（优化后） |
| **能量范围** | 45～100 kVp 诊断范围 | 10～120 keV（CT 诊断范围） |
| **关键局限** | 无理论分析病态性来源 | 未考虑多能道（双探测器）场景 |

---

## 四、本项目能谱重建方案建议

结合本项目的实际情况：
- **设备**：双能 XRT，200～320 kV 高压 X 射线管，Cu/Fe/Al 三材质阶梯标样（共 10 阶，2mm～20mm for Cu/Fe，12mm～30mm for Al）
- **已有代码**：[reconstruct_spectrum.py](file:///e:/photo_electric_II/apd_acd_pipeline/reconstruct_spectrum.py) 中已实现**增广正则化 NNLS（方法一）**，效果良好（APD R² ≈ 0.43）

### 4.1 当前方法的位置定位

本项目已实现的方法一（NNLS 正则化）相当于**Huang 1986 的现代改进版**：
- 同样基于离散化 Fredholm 积分方程，建立系统矩阵 $A$
- 用增广 NNLS 代替 Huang 的迭代展开法，同时引入 Tikhonov 平滑正则化 + 归一化约束 + 边界约束
- 使用三材质（Al + Fe + Cu）联合约束，比 Huang 仅用铝更稳健

### 4.2 针对 Li 2021 论文的改进方向

#### 建议 A：验证当前阶梯厚度是否为"指数分布"

Li 2021 发现最优阶梯排布接近指数序列 $l_i \approx a \cdot e^{bi}$。当前阶梯样参数：

| 材质 | 当前厚度序列（mm） |
|---|---|
| Cu | 2, 4, 6, 8, 10, 12, 14, 16, 18, 20 |
| Fe | 2, 4, 6, 8, 10, 12, 14, 16, 18, 20 |
| Al | 12, 14, 16, 18, 20, 22, 24, 26, 28, 30 |

**结论**：Cu/Fe/Al 均为等差（线性）序列，**恰好是 Li 2021 所批评的最劣排布方式**。这解释了当前 APD $R^2$ 偏低（~0.43）的原因之一。

> [!IMPORTANT]
> 由于物理阶梯已制作完成无法更改，改进方向应侧重于算法侧：使用 Li 2021 的 EM 算法替代或补充 NNLS，或者在正则化参数选取上更加精细化。

#### 建议 B：实现 EM 算法并与 NNLS 对比

**EM 算法实现方案：**

```python
import numpy as np

def reconstruct_spectrum_EM(A, T, energies_keV, n_iter=10000, 
                             char_peaks_keV=None, char_peak_weight=5.0):
    """
    用期望最大化（EM）算法重建 X 射线能谱。
    
    参数：
    - A (np.ndarray): 系统矩阵 (N_meas, N_energy)，A[i,j] = exp(-mu_j * d_i)
    - T (np.ndarray): 实测透射率向量 (N_meas,)，值为 I/I0 ∈ (0, 1]
    - energies_keV (np.ndarray): 能量网格 (N_energy,)，单位 keV
    - n_iter (int): EM 最大迭代次数
    - char_peaks_keV (list): 特征峰位置（如钨靶约 59 keV 和 67 keV），None 则不初始化先验
    - char_peak_weight (float): 特征峰在初始化时的权重倍数（相对于背景值 1.0）
    
    返回：
    - w (np.ndarray): 归一化能谱向量 (N_energy,)
    """
    N_energy = len(energies_keV)
    dE = energies_keV[1] - energies_keV[0]
    
    # 1. 初始化：先验（若已知特征峰位置）
    w = np.ones(N_energy)
    if char_peaks_keV is not None:
        for peak_keV in char_peaks_keV:
            idx = np.argmin(np.abs(energies_keV - peak_keV))
            w[idx] = char_peak_weight
    w = w / (w.sum() * dE)   # 归一化
    
    # 2. 预计算列和（归一化因子）
    col_sum = A.sum(axis=0)   # shape: (N_energy,)
    
    # 3. EM 迭代（乘性更新）
    for it in range(n_iter):
        # 预测透射率
        T_pred = A @ w   # shape: (N_meas,)
        T_pred = np.maximum(T_pred, 1e-12)
        
        # 反向加权
        ratio = T / T_pred   # shape: (N_meas,)
        
        # 更新公式：w_j^(k+1) = w_j^(k) / col_sum_j * sum_i(a_ij * ratio_i)
        correction = (A * ratio[:, np.newaxis]).sum(axis=0)   # shape: (N_energy,)
        w = w * correction / col_sum
        
        # 非负截断（EM 自然保证，此处防浮点数负值）
        w = np.maximum(w, 0.0)
        
        # 归一化（保持 sum(w)*dE = 1）
        w_sum = w.sum() * dE
        if w_sum > 0:
            w = w / w_sum
    
    return w
```

#### 建议 C：组合 NNLS + EM 的两阶段策略

| 阶段 | 方法 | 目的 |
|---|---|---|
| 阶段 1 | **NNLS 正则化**（现有代码）| 快速得到平滑的粗略谱，用作 EM 初值 |
| 阶段 2 | **EM 迭代精修**（新增）| 在 NNLS 谱基础上精细还原特征峰和高频细节 |

**具体流程：**
1. 运行 `reconstruct_channel_spectrum()` → 得到 NNLS 初始谱 $S^{(0)}$
2. 将 $S^{(0)}$ 作为 EM 的初始值运行 EM 迭代（而非全 1 初始化）
3. EM 收敛后输出最终谱 $S^*$，代入 `solve_apd_acd_nonlinear()` 求解

#### 建议 D：根据 Li 2021 的材质选择建议优先使用 Cu/Fe

基于 Li 2021 的条件数分析：

| 条件 | 推荐材质 | 原因 |
|---|---|---|
| 测量点数 $M \leq 7$ | **Al** | 条件数最小 |
| 测量点数 $M \geq 10$ | **Cu 或 Fe** | 条件数远小于 Al |
| 本项目（Cu+Fe+Al 各 10 阶） | **Cu + Fe 为主** | 条件数更优，Al 作为低能约束补充 |

**当前代码改进建议**：在 `load_transmission_data()` 中，可以对 Cu 和 Fe 的阶梯保留所有 10 阶（不裁剪），对 Al 的厚端（>20mm 以上）酌情裁剪，以减少 Al 厚端硬化噪声的干扰。

#### 建议 E：引入混合多材质联合约束

Li 2021 Table 3 显示，Al(0.066cm) + Cu(0.054cm) + Fe(0.001cm) 等多材质混合可将条件数降到 38.6，优于任何单材质。虽然我们的阶梯是固定的，但可以在构建系统矩阵时**选择性地挑选最优的阶梯组合**（例如从 Al 的 10 个阶梯中只取 3～5 个信息量最大的），而不是全部纳入。

### 4.3 推荐实施优先级

```
优先级 1（低风险，立即可做）：
  → 在 reconstruct_spectrum.py 中新增 EM 算法函数 reconstruct_spectrum_EM()
  → 运行与 NNLS 的对比实验，比较 APD/ACD 线性度 R²

优先级 2（中等风险）：
  → 实现两阶段策略：NNLS 初始化 → EM 精修
  → 加入钨靶特征峰先验（59 keV 和 67 keV 特征 K 线）

优先级 3（系统改进）：
  → 对 Cu/Fe 阶梯的厚度设计进行回顾，若未来有新标样制作机会，
    参照 Li 2021 的指数序列 l = a·exp(b·i) 制作更优的阶梯
```

---

## 五、关键公式速查

| 公式 | 含义 |
|---|---|
| $T_i = \sum_j S_j e^{-\mu_j d_i}$ | 正向物理模型（离散化） |
| $\mathbf{A}\mathbf{S} \approx \mathbf{T}$ | 矩阵方程形式 |
| $S_j^{(k+1)} = \frac{S_j^{(k)}}{\sum_i A_{ij}} \sum_i \frac{A_{ij} T_i}{\sum_{j'} A_{ij'} S_{j'}^{(k)}}$ | EM 乘性更新公式 |
| $\text{cond}(A) \sim O(2^N)$ | 等差线性排布的条件数增长规律 |
| $l_i \approx a \cdot e^{bi}$ | Li 2021 发现的最优厚度指数序列 |
| $T'(0) = -\langle\mu\rangle$ | Huang 1986 的等效能量辅助估算 |

---

## 六、参考文献

1. Huang P-H, Chen T-S, Kase KR. Reconstruction of diagnostic x-ray spectra by numerical analysis of transmission data. *Med. Phys.* **13**, 707–710 (1986). [DOI:10.1118/1.595834](https://doi.org/10.1118/1.595834)

2. Li M, Fan F-L, Cong W, Wang G. EM estimation of the x-ray spectrum with a genetically optimized step-wedge phantom. *Front. Phys.* **9**, 678171 (2021). [DOI:10.3389/fphy.2021.678171](https://doi.org/10.3389/fphy.2021.678171)

3. Sidky EY et al. A robust method of x-ray source spectrum estimation from transmission measurements. *J. Appl. Phys.* 97, 124701 (2005).

4. 本项目已有文档：[notes_spectrum_reconstruction.md](file:///e:/photo_electric_II/paper/notes_spectrum_reconstruction.md)
