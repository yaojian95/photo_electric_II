"""
矿石密度与品位关系深度分析脚本
=================================
数据字段:
    矿石编号: 矿石唯一标识
    Cu (%)  : 铜的重量百分比品位
    Fe (%)  : 铁的重量百分比品位
    S  (%)  : 硫的重量百分比品位
    质量 (g): 矿石质量
    密度(g/ml): 测量密度

分析维度:
    1. 密度物理模型与品位相关性
    2. 多元回归残差分析
    3. 四类典型异常场景识别
    4. 矿石分选意义评估
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os

# --- 字体配置 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 输出目录 ---
OUTPUT_DIR = 'analysis_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===========================================================
# 数据加载与预处理
# ===========================================================
df = pd.read_excel('工作簿1.xlsx')
col_id      = '矿石编号'
col_cu      = 'Cu'
col_fe      = 'Fe'
col_s       = 'S'
col_mass    = '质量'
col_density = '密度(g/ml)'
features    = [col_cu, col_fe, col_s]

# 计算"综合金属品位" —— 物理等权重（Cu + Fe + S）
# 采用等权重加和，代表矿石中金属/硫化物组分的总化学含量（质量百分比之和）。
# 不引入经济价格权重，以保持物理中性；若需按经济价值分析，应另行计算。
df['总金属品位'] = df[col_cu] + df[col_fe] + df[col_s]

# ===========================================================
# 0. 密度分段统计（质量分布 + 质量加权品位）
# ===========================================================
# 分段边界（对应分选意义划分）
bins   = [0, 2.5, 2.7, 3.0, 3.2, 10.0]
labels = ['<2.5', '2.5~2.7', '2.7~3.0', '3.0~3.2', '>3.2']
df['密度段'] = pd.cut(df[col_density], bins=bins, labels=labels, right=True)

seg_stats = []
for seg in labels:
    sub = df[df['密度段'] == seg]
    if len(sub) == 0:
        seg_stats.append({'密度段': seg, '样本数': 0, '总质量_g': 0,
                          '质量占比_%': 0, '质量加权Cu_%': np.nan,
                          '质量加权Fe_%': np.nan, '质量加权S_%': np.nan})
        continue
    total_mass = sub[col_mass].sum()
    seg_stats.append({
        '密度段':       seg,
        '样本数':       len(sub),
        '总质量_g':     round(total_mass, 1),
        '质量占比_%':   round(total_mass / df[col_mass].sum() * 100, 2),
        # 质量加权品位 = sum(质量_i × 品位_i) / sum(质量_i)
        '质量加权Cu_%': round((sub[col_mass] * sub[col_cu]).sum() / total_mass, 4),
        '质量加权Fe_%': round((sub[col_mass] * sub[col_fe]).sum() / total_mass, 3),
        '质量加权S_%':  round((sub[col_mass] * sub[col_s ]).sum() / total_mass, 3),
    })

df_seg = pd.DataFrame(seg_stats)
df_seg.to_csv(os.path.join(OUTPUT_DIR, 'density_segment_stats.csv'), index=False, encoding='utf-8-sig')

print('\n=== 密度分段统计 ===')
print(df_seg.to_string(index=False))

# --- 图0a: 各密度段质量占比柱状图
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
colors_seg = ['#74b9ff', '#a29bfe', '#55efc4', '#fdcb6e', '#e17055']

axes[0].bar(df_seg['密度段'], df_seg['质量占比_%'], color=colors_seg, edgecolor='k', linewidth=0.7)
for i, (v, n) in enumerate(zip(df_seg['质量占比_%'], df_seg['样本数'])):
    axes[0].text(i, v + 0.3, f'{v:.1f}%\n(n={n})', ha='center', va='bottom', fontsize=9)
axes[0].set_xlabel('密度段 (g/ml)', fontsize=12)
axes[0].set_ylabel('质量占比 (%)', fontsize=12)
axes[0].set_title('各密度段矿石质量占比', fontsize=13, fontweight='bold')
axes[0].set_ylim(0, df_seg['质量占比_%'].max() * 1.2)

# --- 图0b: 各密度段质量加权品位
x = np.arange(len(labels))
w = 0.25
axes[1].bar(x - w, df_seg['质量加权Cu_%'], width=w, label='Cu (%)', color='#e17055', edgecolor='k', lw=0.6)
axes[1].bar(x,     df_seg['质量加权Fe_%'], width=w, label='Fe (%)', color='#74b9ff', edgecolor='k', lw=0.6)
axes[1].bar(x + w, df_seg['质量加权S_%'],  width=w, label='S (%)',  color='#fdcb6e', edgecolor='k', lw=0.6)
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels)
axes[1].set_xlabel('密度段 (g/ml)', fontsize=12)
axes[1].set_ylabel('质量加权品位 (%)', fontsize=12)
axes[1].set_title('各密度段质量加权元素品位', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)

plt.suptitle('矿石密度分段统计分析', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '00_density_segment_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()

# ===========================================================
# 1. 相关性热力图 (增强版)
# ===========================================================
corr_cols = [col_density, col_cu, col_fe, col_s, '总金属品位']
corr_matrix = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
            vmin=-1, vmax=1, ax=ax, linewidths=0.5, annot_kws={'size': 12})
ax.set_title('矿石密度与品位相关性热力图（皮尔逊系数）', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '01_correlation_heatmap.png'), dpi=150)
plt.close()

# ===========================================================
# 2. 多元线性回归 + 残差分析
# ===========================================================
X = df[features].values
y = df[col_density].values
lr = LinearRegression()
lr.fit(X, y)
y_pred = lr.predict(X)
residuals = y - y_pred
std_resid = np.std(residuals)
z_scores = residuals / std_resid
r2 = lr.score(X, y)

df['预测密度']    = y_pred
df['残差']        = residuals
df['残差Z分数']   = z_scores

print(f"\n=== 线性回归模型 ===")
print(f"  Cu系数: {lr.coef_[0]:.4f}")
print(f"  Fe系数: {lr.coef_[1]:.4f}")
print(f"  S 系数: {lr.coef_[2]:.4f}")
print(f"  截距  : {lr.intercept_:.4f}")
print(f"  R2    : {r2:.4f}")

# ===========================================================
# 3. 孤立森林 (全特征)
# ===========================================================
scaler = StandardScaler()
X_all = df[[col_density, col_cu, col_fe, col_s]].values
X_scaled = scaler.fit_transform(X_all)
iso = IsolationForest(contamination=0.08, random_state=42)
df['孤立森林'] = iso.fit_predict(X_scaled)  # -1=异常, 1=正常

# ===========================================================
# 4. 四类典型异常场景定义
# ===========================================================
# 计算关键分位数
cu_p75  = df[col_cu].quantile(0.75)
fe_p75  = df[col_fe].quantile(0.75)
rho_p25 = df[col_density].quantile(0.25)
rho_p75 = df[col_density].quantile(0.75)

# 【场景A】高金属品位 + 低密度 → 最典型负样本（密度测量偏低 或 大孔隙）
mask_A = (
    ((df[col_cu] > cu_p75) | (df[col_fe] > fe_p75)) &
    (df[col_density] < rho_p25)
)

# 【场景B】低金属品位 + 高密度 → 可能含有稠密非金属矿物（重晶石、钛铁矿等）
mask_B = (
    (df[col_cu] < df[col_cu].quantile(0.25)) &
    (df[col_fe] < df[col_fe].quantile(0.25)) &
    (df[col_density] > rho_p75)
)

# 【场景C】密度残差Z分数 > +2（密度远高于品位预测值）
mask_C = df['残差Z分数'] >  2.0

# 【场景D】密度残差Z分数 < -2（密度远低于品位预测值）
mask_D = df['残差Z分数'] < -2.0

df['异常类型'] = '正常'
df.loc[mask_A, '异常类型'] = 'A: 高品位低密度'
df.loc[mask_B, '异常类型'] = 'B: 低品位高密度'
df.loc[mask_C & ~mask_A & ~mask_B, '异常类型'] = 'C: 密度异常偏高'
df.loc[mask_D & ~mask_A & ~mask_B, '异常类型'] = 'D: 密度异常偏低'

# 保存所有异常样本
any_anomaly = mask_A | mask_B | mask_C | mask_D
df_anomaly = df[any_anomaly].copy()
df_anomaly.to_csv(os.path.join(OUTPUT_DIR, 'all_anomaly_samples.csv'),
                  index=False, encoding='utf-8-sig')

# ===========================================================
# 5. 主要可视化图表
# ===========================================================

# --- 图1: 实际密度 vs 预测密度（四类异常标注）
colors_map = {
    '正常':         '#a8d8ea',
    'A: 高品位低密度': '#e74c3c',
    'B: 低品位高密度': '#f39c12',
    'C: 密度异常偏高': '#8e44ad',
    'D: 密度异常偏低': '#27ae60',
}
fig, ax = plt.subplots(figsize=(9, 7))
for label, color in colors_map.items():
    sub = df[df['异常类型'] == label]
    marker = 'o' if label == '正常' else (
        'v' if 'A' in label else ('s' if 'B' in label else ('^' if 'C' in label else 'D'))
    )
    ax.scatter(sub['预测密度'], sub[col_density], label=f'{label} (n={len(sub)})',
               c=color, s=60 if label == '正常' else 120,
               edgecolors='k', linewidths=0.5 if label == '正常' else 1.2,
               alpha=0.75, marker=marker, zorder=3 if label != '正常' else 2)

# 标注A类异常样本号
for _, row in df[mask_A].iterrows():
    ax.annotate(str(row[col_id])[-4:], (row['预测密度'], row[col_density]),
                fontsize=7.5, color='#c0392b',
                xytext=(4, 4), textcoords='offset points')

lims = [2.0, 4.6]
ax.plot(lims, lims, 'k--', lw=1.2, label='理想线 y=x', alpha=0.6)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('基于品位预测的密度 (g/ml)', fontsize=12)
ax.set_ylabel('实测密度 (g/ml)', fontsize=12)
ax.set_title('实测密度 vs 回归预测密度\n（异常类型分类标注）', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
ax.text(0.98, 0.03, f'R² = {r2:.3f}\n截距 = {lr.intercept_:.3f}',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '02_density_pred_vs_actual.png'), dpi=150)
plt.close()

# --- 图2: 密度 vs Cu / Fe / S 的散点图（分类着色）
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
feat_labels = [f'Cu 品位 (%)', f'Fe 品位 (%)', f'S 品位 (%)']
for i, (feat, xlabel) in enumerate(zip(features, feat_labels)):
    ax = axes[i]
    for label, color in colors_map.items():
        sub = df[df['异常类型'] == label]
        ax.scatter(sub[feat], sub[col_density],
                   c=color, label=label, s=50 if label == '正常' else 90,
                   edgecolors='k', linewidths=0.4 if label == '正常' else 1,
                   alpha=0.75, zorder=3 if label != '正常' else 2)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel('密度 (g/ml)', fontsize=11)
    ax.set_title(f'密度 vs {feat}', fontsize=12, fontweight='bold')
    if i == 0:
        ax.legend(fontsize=7.5)
plt.suptitle('矿石密度与各元素品位关系（分类异常标注）', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '03_density_vs_features.png'), dpi=150, bbox_inches='tight')
plt.close()

# --- 图3: 密度分布直方图与正态拟合
from scipy import stats
fig, ax = plt.subplots(figsize=(8, 5))
normal_data = df[df['异常类型'] == '正常'][col_density]
ax.hist(df[col_density], bins=20, alpha=0.4, color='steelblue', label='全部样本', density=True)
ax.hist(normal_data, bins=20, alpha=0.6, color='seagreen', label='正常样本', density=True)
mu, sigma = stats.norm.fit(df[col_density])
x = np.linspace(1.8, 5.0, 200)
ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2, label=f'正态拟合 μ={mu:.2f} σ={sigma:.2f}')
ax.axvline(rho_p25, color='orange', ls='--', lw=1.5, label=f'Q25={rho_p25:.2f}')
ax.axvline(rho_p75, color='purple', ls='--', lw=1.5, label=f'Q75={rho_p75:.2f}')
ax.set_xlabel('密度 (g/ml)', fontsize=12)
ax.set_ylabel('频率密度', fontsize=12)
ax.set_title('矿石密度分布直方图', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '04_density_histogram.png'), dpi=150)
plt.close()

# --- 图4: 总金属品位 vs 密度 + 二维KDE轮廓
fig, ax = plt.subplots(figsize=(9, 6))
norm_df = df[df['异常类型'] == '正常']
sns.kdeplot(x=norm_df['总金属品位'], y=norm_df[col_density], ax=ax,
            fill=True, alpha=0.3, cmap='Blues', levels=8)
for label, color in colors_map.items():
    sub = df[df['异常类型'] == label]
    if len(sub) == 0: continue
    ax.scatter(sub['总金属品位'], sub[col_density],
               c=color, s=80 if label == '正常' else 130,
               edgecolors='k', linewidths=0.5 if label == '正常' else 1.5,
               alpha=0.85, label=f'{label} (n={len(sub)})', zorder=4)
ax.set_xlabel('综合金属品位 (Cu + Fe + S) %', fontsize=12)
ax.set_ylabel('密度 (g/ml)', fontsize=12)
ax.set_title('综合金属品位 vs 密度\n（等权物理品位 | KDE密度轮廓 + 异常点标注）', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '05_grade_vs_density_kde.png'), dpi=150)
plt.close()

# ===========================================================
# 6. 控制台汇总输出
# ===========================================================
print("\n=== 异常样本汇总 ===")
for label in ['A: 高品位低密度', 'B: 低品位高密度', 'C: 密度异常偏高', 'D: 密度异常偏低']:
    sub = df[df['异常类型'] == label]
    print(f"\n【{label}】 共 {len(sub)} 个:")
    for _, row in sub.iterrows():
        print(f"  #{row[col_id]}  Cu={row[col_cu]:.3f}%  Fe={row[col_fe]:.2f}%  "
              f"S={row[col_s]:.2f}%  密度={row[col_density]:.3f}g/ml  "
              f"残差Z={row['残差Z分数']:.2f}")

print("\n所有分析图表已保存至:", OUTPUT_DIR)
print(f"异常样本CSV已保存: {os.path.join(OUTPUT_DIR, 'all_anomaly_samples.csv')}")
