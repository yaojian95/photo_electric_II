"""
生成合并版 00~10 图像（combined batch: 0325 + 0520）
+ Cu/Fe 联合分选曲线（06c/07c/08c）
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy import stats
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = 'analysis_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. 数据加载（复用 analyze_combined 的逻辑）
# ============================================================
df_325 = pd.read_excel('工作簿1.xlsx', sheet_name='0325')
df_325.columns = ['矿石编号', 'Cu', 'Fe', 'S', '质量', '密度']
df_325['批次'] = '0325'

df_520 = pd.read_excel('工作簿1.xlsx', sheet_name='Sheet1')
df_520 = df_520.rename(columns={
    df_520.columns[1]: '质量',
    df_520.columns[2]: '矿石编号',
    'density': '密度',
})
df_520 = df_520[['矿石编号', 'Cu', 'Fe', '质量', '密度']].copy()
df_520['密度'] = pd.to_numeric(df_520['密度'], errors='coerce')
df_520_clean = df_520[(df_520['密度'] >= 1.0) & (df_520['密度'] <= 6.0)].copy()
df_520_clean['S'] = np.nan
df_520_clean['批次'] = '0520'

df = pd.concat([df_325, df_520_clean[['矿石编号','Cu','Fe','S','质量','密度','批次']]],
               ignore_index=True)

col_mass = '质量'
col_density = '密度'
col_cu = 'Cu'; col_fe = 'Fe'; col_s = 'S'
features_reg = ['Cu', 'Fe']   # S 缺失，只用 Cu/Fe 做回归
colors_batch = {'0325': '#e17055', '0520': '#74b9ff'}

M_feed = df[col_mass].sum()
alpha_cu = (df[col_mass] * df[col_cu]).sum() / M_feed
alpha_fe = (df[col_mass] * df[col_fe]).sum() / M_feed
alpha_s  = (df[col_mass].loc[df[col_s].notna()] * df[col_s].loc[df[col_s].notna()]).sum() / M_feed

# 线性回归（Cu + Fe）
X = df[features_reg].values
y = df[col_density].values
lr = LinearRegression().fit(X, y)
y_pred = lr.predict(X)
residuals = y - y_pred
std_r = np.std(residuals)
df['预测密度'] = y_pred
df['残差Z'] = residuals / std_r
r2 = lr.score(X, y)

# 四类异常分类（场景A/C/D，场景B本数据为0）
cu_p75 = df[col_cu].quantile(0.75)
fe_p75 = df[col_fe].quantile(0.75)
rho_p25 = df[col_density].quantile(0.25)
rho_p75 = df[col_density].quantile(0.75)

mask_A = ((df[col_cu] > cu_p75) | (df[col_fe] > fe_p75)) & (df[col_density] < rho_p25)
mask_C = df['残差Z'] > 2.0
mask_D = df['残差Z'] < -2.0
df['异常类型'] = '正常'
df.loc[mask_A, '异常类型'] = 'A: 高品位低密度'
df.loc[mask_C & ~mask_A, '异常类型'] = 'C: 密度异常偏高'
df.loc[mask_D & ~mask_A, '异常类型'] = 'D: 密度异常偏低'

colors_map = {
    '正常':         '#a8d8ea',
    'A: 高品位低密度': '#e74c3c',
    'C: 密度异常偏高': '#8e44ad',
    'D: 密度异常偏低': '#27ae60',
}

# 密度分段
bins   = [0, 2.5, 2.7, 3.0, 3.2, 10.0]
labels = ['<2.5', '2.5~2.7', '2.7~3.0', '3.0~3.2', '>3.2']
df['密度段'] = pd.cut(df[col_density], bins=bins, labels=labels)

# ============================================================
# 图 00c: 各密度段质量占比 + 质量加权品位（合并版）
# ============================================================
seg_stats = []
for seg in labels:
    sub = df[df['密度段'] == seg]
    if len(sub) == 0:
        continue
    m = sub[col_mass].sum()
    seg_stats.append({
        '密度段': seg, '样本数': len(sub),
        '质量(g)': m,
        '质量占比%': m / M_feed * 100,
        '质量加权Cu%': (sub[col_mass] * sub[col_cu]).sum() / m,
        '质量加权Fe%': (sub[col_mass] * sub[col_fe]).sum() / m,
    })
df_seg = pd.DataFrame(seg_stats)

seg_colors = ['#74b9ff', '#a29bfe', '#55efc4', '#fdcb6e', '#e17055']
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].bar(df_seg['密度段'], df_seg['质量占比%'], color=seg_colors[:len(df_seg)], edgecolor='k', lw=0.7)
for i, row in df_seg.iterrows():
    axes[0].text(i, row['质量占比%'] + 0.3, f"{row['质量占比%']:.1f}%\n(n={row['样本数']})",
                 ha='center', va='bottom', fontsize=9)
axes[0].set_xlabel('密度段 (g/ml)', fontsize=12); axes[0].set_ylabel('质量占比 (%)', fontsize=12)
axes[0].set_title('各密度段质量占比（合并）', fontsize=12, fontweight='bold')
axes[0].set_ylim(0, df_seg['质量占比%'].max() * 1.25)

axes[1].bar(df_seg['密度段'], df_seg['质量加权Cu%'], color='#e17055', edgecolor='k', lw=0.7)
axes[1].axhline(alpha_cu, color='k', ls='--', lw=1.2, label=f'给矿均值 {alpha_cu:.4f}%')
axes[1].set_xlabel('密度段 (g/ml)', fontsize=12); axes[1].set_ylabel('质量加权 Cu 品位 (%)', fontsize=12)
axes[1].set_title('各密度段质量加权 Cu 品位', fontsize=12, fontweight='bold'); axes[1].legend()

axes[2].bar(df_seg['密度段'], df_seg['质量加权Fe%'], color='#74b9ff', edgecolor='k', lw=0.7)
axes[2].axhline(alpha_fe, color='k', ls='--', lw=1.2, label=f'给矿均值 {alpha_fe:.3f}%')
axes[2].set_xlabel('密度段 (g/ml)', fontsize=12); axes[2].set_ylabel('质量加权 Fe 品位 (%)', fontsize=12)
axes[2].set_title('各密度段质量加权 Fe 品位', fontsize=12, fontweight='bold'); axes[2].legend()

plt.suptitle('合并批次（0325+0520）各密度段统计', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '00c_density_segment_combined.png'), dpi=150, bbox_inches='tight')
plt.close(); print('00c 完成')

# ============================================================
# 图 01c: 相关系数热力图（合并版，包含 S 与总金属品位）
# ============================================================
df['总金属品位'] = df['Cu'] + df['Fe'] + df['S']
corr_cols = ['密度', 'Cu', 'Fe', 'S', '总金属品位']
df_corr = df[corr_cols].copy()
corr_matrix = df_corr.corr()

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
            vmin=-1, vmax=1, center=0, linewidths=0.5, ax=ax,
            xticklabels=corr_cols, yticklabels=corr_cols)
ax.set_title('相关系数热力图（合并批次，密度/Cu/Fe/S/总金属品位）', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '01c_correlation_heatmap_combined.png'), dpi=150)
plt.close(); print('01c 完成')

# ============================================================
# 图 02c: 实测密度 vs 预测密度（两批次着色 + 异常标注）
# ============================================================
fig, ax = plt.subplots(figsize=(9, 7))
for bat, color in colors_batch.items():
    sub = df[df['批次'] == bat]
    ax.scatter(sub['预测密度'], sub['密度'], c=color, s=55,
               edgecolors='k', lw=0.4, alpha=0.75, label=bat, zorder=2)
# 异常点高亮
for label, color in colors_map.items():
    if label == '正常': continue
    sub = df[df['异常类型'] == label]
    if len(sub) == 0: continue
    ax.scatter(sub['预测密度'], sub['密度'], c=color, s=120,
               edgecolors='k', lw=1.2, alpha=0.9, label=f'{label} (n={len(sub)})', zorder=3, marker='^')
# 极端点标注
for _, row in df[df['残差Z'].abs() > 3.0].iterrows():
    ax.annotate(str(row['矿石编号'])[-6:], (row['预测密度'], row['密度']),
                fontsize=7, color='red', xytext=(5, 3), textcoords='offset points')
lims = [1.5, 5.0]
ax.plot(lims, lims, 'k--', lw=1.2, alpha=0.6, label='理想线 y=x')
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('Cu+Fe 预测密度 (g/ml)', fontsize=12); ax.set_ylabel('实测密度 (g/ml)', fontsize=12)
ax.set_title(f'实测密度 vs 预测密度（Cu+Fe 回归，R²={r2:.3f}）\n异常类型分类标注', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '02c_density_pred_vs_actual_combined.png'), dpi=150)
plt.close(); print('02c 完成')

# ============================================================
# 图 03c: 密度 vs Cu / Fe（两批次着色）
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for bat, color in colors_batch.items():
    sub = df[df['批次'] == bat]
    axes[0].scatter(sub['Cu'], sub['密度'], c=color, s=45, alpha=0.7,
                    edgecolors='k', lw=0.4, label=bat)
    axes[1].scatter(sub['Fe'], sub['密度'], c=color, s=45, alpha=0.7,
                    edgecolors='k', lw=0.4, label=bat)
for ax, col, xlabel in zip(axes, ['Cu', 'Fe'], ['Cu 品位 (%)', 'Fe 品位 (%)']):
    r = df[col].corr(df['密度'])
    ax.set_xlabel(xlabel, fontsize=12); ax.set_ylabel('密度 (g/ml)', fontsize=12)
    ax.set_title(f'密度 vs {xlabel}（全体 r={r:.3f}）', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
plt.suptitle('密度与各元素品位关系（合并批次）', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '03c_density_vs_features_combined.png'), dpi=150, bbox_inches='tight')
plt.close(); print('03c 完成')

# ============================================================
# 图 04c: 密度分布直方图（两批次对比）
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5))
for bat, color in colors_batch.items():
    sub = df[df['批次'] == bat]['密度']
    ax.hist(sub, bins=25, alpha=0.5, color=color, label=bat,
            edgecolor='k', lw=0.4, density=True)
    ax.axvline(sub.mean(), color=color, ls='--', lw=1.8, label=f'{bat} 均值={sub.mean():.2f}')
mu, sigma = stats.norm.fit(df['密度'])
x = np.linspace(1.5, 5.0, 200)
ax.plot(x, stats.norm.pdf(x, mu, sigma), 'k-', lw=1.8, label=f'全体正态拟合 μ={mu:.2f} σ={sigma:.2f}')
ax.set_xlabel('密度 (g/ml)', fontsize=12); ax.set_ylabel('频率密度', fontsize=12)
ax.set_title('密度分布直方图（两批次对比）', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '04c_density_histogram_combined.png'), dpi=150)
plt.close(); print('04c 完成')

# ============================================================
# 图 05c: Cu+Fe 综合品位 vs 密度 KDE（两批次子图）
# ============================================================
df['综合Cu+Fe'] = df['Cu'] + df['Fe']
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, bat in zip(axes, ['0325', '0520']):
    sub = df[df['批次'] == bat]
    try:
        sns.kdeplot(x=sub['综合Cu+Fe'], y=sub['密度'], ax=ax,
                    fill=True, alpha=0.35, cmap='Blues', levels=6)
    except Exception:
        pass
    ax.scatter(sub['综合Cu+Fe'], sub['密度'], c=colors_batch[bat], s=45, alpha=0.7,
               edgecolors='k', lw=0.4)
    extreme = sub[sub['残差Z'].abs() > 2.5]
    for _, row in extreme.iterrows():
        ax.annotate(str(row['矿石编号'])[-6:], (row['综合Cu+Fe'], row['密度']),
                    fontsize=7, color='red', xytext=(4, 3), textcoords='offset points')
    ax.set_xlabel('Cu + Fe 品位 (%)', fontsize=12); ax.set_ylabel('密度 (g/ml)', fontsize=12)
    ax.set_title(f'{bat} 批次：Cu+Fe 品位-密度 KDE', fontsize=12, fontweight='bold')
plt.suptitle('综合品位(Cu+Fe) vs 密度 KDE（两批次分开）', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '05c_grade_density_kde_combined.png'), dpi=150, bbox_inches='tight')
plt.close(); print('05c 完成')

# ============================================================
# 图 06c/07c: 分选曲线（Cu + Fe 同图）— 合并 + 0325 单批次
# ============================================================
def calc_sorting_scan(df_in, n_pts=500):
    """全局密度阈值扫描，返回 DataFrame，计算 Cu 和 Fe 的各分选指标。
    
    参数：
        df_in : pd.DataFrame，含 Cu, Fe, 质量, 密度 列
        n_pts : int，扫描点数
    
    返回：
        pd.DataFrame，每行对应一个阈值的分选指标
    """
    M = df_in['质量'].sum()
    a_cu = (df_in['质量'] * df_in['Cu']).sum() / M
    a_fe = (df_in['质量'] * df_in['Fe']).sum() / M
    thrs = np.linspace(df_in['密度'].min(), df_in['密度'].max(), n_pts)
    recs = []
    for t in thrs:
        c = df_in[df_in['密度'] > t]
        tl = df_in[df_in['密度'] <= t]
        if len(c) == 0:
            recs.append({'rho_t': t, 'yield_pct': 0, 'rec_cu': 0, 'rec_fe': 0,
                         'enrich_cu': np.nan, 'enrich_fe': np.nan})
            continue
        Mc = c['质量'].sum()
        b_cu = (c['质量'] * c['Cu']).sum() / Mc
        b_fe = (c['质量'] * c['Fe']).sum() / Mc
        yld = Mc / M * 100
        recs.append({'rho_t': t, 'yield_pct': yld,
                     'rec_cu': Mc * b_cu / (M * a_cu) * 100,
                     'rec_fe': Mc * b_fe / (M * a_fe) * 100,
                     'enrich_cu': b_cu / a_cu,
                     'enrich_fe': b_fe / a_fe,
                     'beta_cu': b_cu, 'beta_fe': b_fe})
    return pd.DataFrame(recs), a_cu, a_fe

def plot_sorting_pair(res, a_cu, a_fe, title_prefix, suffix, typical_thrs=None):
    """绘制 Cu+Fe 联合分选曲线（选别曲线 + 富集比 + 品位 vs 阈值），保存 06c/07c/08c。
    
    参数：
        res          : pd.DataFrame，calc_sorting_scan 的返回值
        a_cu         : float，给矿 Cu 品位
        a_fe         : float，给矿 Fe 品位
        title_prefix : str，图标题前缀
        suffix       : str，文件名后缀（如 'c_comb'）
        typical_thrs : list of (label, rho_t)，典型阈值标注点
    """
    # ---- 图 06c: 选别曲线（产率 vs 回收率）----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.plot(res['yield_pct'], res['rec_cu'], color='#e17055', lw=2.2, label='Cu 回收率')
    ax.plot(res['yield_pct'], res['rec_fe'], color='#74b9ff', lw=2.2, label='Fe 回收率')
    ax.plot(res['yield_pct'], res['yield_pct'], 'k--', lw=1.2, alpha=0.4, label='γ=ε 无富集线')
    ax.set_xlabel('产率 γ (%)', fontsize=12); ax.set_ylabel('回收率 ε (%)', fontsize=12)
    ax.set_title('选别曲线：产率 vs 回收率', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10); ax.set_xlim(0, 100); ax.set_ylim(0, 105); ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(res['yield_pct'], res['enrich_cu'].replace([np.inf], np.nan), color='#e17055', lw=2.2, label='Cu 富集比')
    ax2.plot(res['yield_pct'], res['enrich_fe'].replace([np.inf], np.nan), color='#74b9ff', lw=2.2, label='Fe 富集比')
    ax2.axhline(1.0, color='k', ls='--', lw=1.2, alpha=0.4)
    ax2.set_xlabel('产率 γ (%)', fontsize=12); ax2.set_ylabel('富集比 k', fontsize=12)
    ax2.set_title('富集比 vs 产率', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10); ax2.set_xlim(0, 100); ax2.grid(True, alpha=0.3)
    ymax = max(res['enrich_cu'].replace([np.inf], np.nan).dropna().quantile(0.95) * 1.3, 5)
    ax2.set_ylim(0, ymax)

    plt.suptitle(f'{title_prefix} — 选别曲线', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'06{suffix}_sorting_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'06{suffix} 完成')

    # ---- 图 07c: 各指标 vs 密度阈值 ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(res['rho_t'], res['yield_pct'], color='steelblue', lw=2)
    ax.set_xlabel('密度阈值 ρ_t (g/ml)', fontsize=11); ax.set_ylabel('产率 γ (%)', fontsize=11)
    ax.set_title('产率 vs 密度阈值', fontsize=12, fontweight='bold'); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(res['rho_t'], res['rec_cu'], color='#e17055', lw=2, label='Cu')
    ax.plot(res['rho_t'], res['rec_fe'], color='#74b9ff', lw=2, label='Fe')
    ax.set_xlabel('密度阈值 ρ_t (g/ml)', fontsize=11); ax.set_ylabel('回收率 ε (%)', fontsize=11)
    ax.set_title('各元素回收率 vs 密度阈值', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(res['rho_t'], res['beta_cu'].replace([np.inf], np.nan), color='#e17055', lw=2, label='精矿 Cu 品位')
    ax.axhline(a_cu, color='#e17055', ls='--', lw=1.2, alpha=0.6, label=f'给矿 Cu={a_cu:.4f}%')
    ax.set_xlabel('密度阈值 ρ_t (g/ml)', fontsize=11); ax.set_ylabel('精矿 Cu 品位 (%)', fontsize=11)
    ax.set_title('精矿 Cu 品位 vs 密度阈值', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(res['rho_t'], res['enrich_cu'].replace([np.inf], np.nan), color='#e17055', lw=2, label='Cu 富集比')
    ax.plot(res['rho_t'], res['enrich_fe'].replace([np.inf], np.nan), color='#74b9ff', lw=2, label='Fe 富集比')
    ax.axhline(1.0, color='k', ls='--', lw=1.2, alpha=0.5)
    ax.set_xlabel('密度阈值 ρ_t (g/ml)', fontsize=11); ax.set_ylabel('富集比 k', fontsize=11)
    ax.set_title('Cu/Fe 富集比 vs 密度阈值', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 12); ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

    plt.suptitle(f'{title_prefix} — 分选指标 vs 阈值', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'07{suffix}_metrics_vs_threshold.png'), dpi=150)
    plt.close()
    print(f'07{suffix} 完成')

    # ---- 图 08c: 典型阈值柱状对比（Cu + Fe 并排）----
    if typical_thrs is None:
        return
    names_short, vals = [], {'yield': [], 'rec_cu': [], 'rec_fe': [], 'enrich_cu': [], 'enrich_fe': [], 'beta_cu': [], 'beta_fe': []}
    for lbl, thr in typical_thrs:
        row_thr = res.iloc[(res['rho_t'] - thr).abs().argsort()[:1]]
        names_short.append(lbl)
        vals['yield'].append(float(row_thr['yield_pct']))
        vals['rec_cu'].append(float(row_thr['rec_cu']))
        vals['rec_fe'].append(float(row_thr['rec_fe']))
        vals['enrich_cu'].append(float(row_thr['enrich_cu']))
        vals['enrich_fe'].append(float(row_thr['enrich_fe']))
        vals['beta_cu'].append(float(row_thr['beta_cu']))
        vals['beta_fe'].append(float(row_thr['beta_fe']))

    x = np.arange(len(names_short))
    w = 0.22
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 产率 + Cu回收率 + Fe回收率
    ax = axes[0]
    b1 = ax.bar(x - w,   vals['yield'],  width=w, label='产率 γ',    color='#636e72', edgecolor='k', lw=0.6, alpha=0.85)
    b2 = ax.bar(x,       vals['rec_cu'], width=w, label='Cu 回收率', color='#e17055', edgecolor='k', lw=0.6, alpha=0.85)
    b3 = ax.bar(x + w,   vals['rec_fe'], width=w, label='Fe 回收率', color='#74b9ff', edgecolor='k', lw=0.6, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(names_short, fontsize=9)
    ax.set_ylabel('%', fontsize=12); ax.set_title('产率 & 回收率对比', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 120); ax.legend(fontsize=9)
    for b in list(b1) + list(b2) + list(b3):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.0,
                f'{b.get_height():.1f}', ha='center', va='bottom', fontsize=7.5)

    # Cu/Fe 富集比
    ax = axes[1]
    b1 = ax.bar(x - w/2, vals['enrich_cu'], width=w, label='Cu 富集比', color='#e17055', edgecolor='k', lw=0.6, alpha=0.85)
    b2 = ax.bar(x + w/2, vals['enrich_fe'], width=w, label='Fe 富集比', color='#74b9ff', edgecolor='k', lw=0.6, alpha=0.85)
    ax.axhline(1.0, color='k', ls='--', lw=1.2, alpha=0.5, label='无富集基准线')
    ax.set_xticks(x); ax.set_xticklabels(names_short, fontsize=9)
    ax.set_ylabel('富集比 k', fontsize=12); ax.set_title('Cu / Fe 富集比对比', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    for b, v in zip(list(b1) + list(b2), vals['enrich_cu'] + vals['enrich_fe']):
        ax.text(b.get_x() + b.get_width()/2, v + 0.05, f'{v:.2f}x', ha='center', va='bottom', fontsize=8)

    # 精矿品位
    ax = axes[2]
    b1 = ax.bar(x - w/2, vals['beta_cu'], width=w, label='精矿 Cu 品位', color='#e17055', edgecolor='k', lw=0.6, alpha=0.85)
    b2 = ax.bar(x + w/2, [v/10 for v in vals['beta_fe']], width=w, label='精矿 Fe 品位 (÷10)', color='#74b9ff', edgecolor='k', lw=0.6, alpha=0.85)
    ax.axhline(a_cu, color='#e17055', ls='--', lw=1, alpha=0.6, label=f'给矿 Cu {a_cu:.3f}%')
    ax.axhline(a_fe/10, color='#74b9ff', ls='--', lw=1, alpha=0.6, label=f'给矿 Fe/10 {a_fe/10:.2f}%')
    ax.set_xticks(x); ax.set_xticklabels(names_short, fontsize=9)
    ax.set_ylabel('精矿品位 (%)  [Fe÷10 显示]', fontsize=11)
    ax.set_title('精矿品位对比', fontsize=12, fontweight='bold'); ax.legend(fontsize=8)
    for b, v in zip(list(b1), vals['beta_cu']):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=7.5)
    for b, v in zip(list(b2), vals['beta_fe']):
        ax.text(b.get_x() + b.get_width()/2, v/10 + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=7.5)

    plt.suptitle(f'{title_prefix} — 典型阈值方案对比', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'08{suffix}_threshold_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'08{suffix} 完成')


# --- 运行：0325 单批次（Cu + Fe + S 原始数据，使用 density 列）---
res_325, a_cu_325, a_fe_325 = calc_sorting_scan(df_325.rename(columns={'密度': '密度', '质量': '质量'}))
plot_sorting_pair(res_325, a_cu_325, a_fe_325,
                  '0325 批次（仅用 Cu+Fe 扫描）', 'a',
                  [('T1\n>2.5', 2.5), ('T2\n>2.7', 2.7), ('T3\n>3.0', 3.0), ('T4\n>3.2', 3.2)])

# --- 运行：合并批次 ---
res_all, a_cu_all, a_fe_all = calc_sorting_scan(df)
plot_sorting_pair(res_all, a_cu_all, a_fe_all,
                  '合并批次（0325+0520）', 'c',
                  [('T1\n>2.0', 2.0), ('T2\n>2.5', 2.5), ('T3\n>2.7', 2.7), ('T4\n>3.0', 3.0)])

print('\n全部图像生成完毕。')
