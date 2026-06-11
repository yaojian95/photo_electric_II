"""
品位分布可视化脚本
==================
生成各元素品位分布直方图 + 箱线图 + 质量加权标注
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = 'analysis_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_excel('工作簿1.xlsx')
col_mass    = '质量'
col_density = '密度(g/ml)'
M = df[col_mass].sum()

elements = {
    'Cu': {'color': '#e17055', 'unit': '%', 'log': True,  'bins': 30},
    'Fe': {'color': '#74b9ff', 'unit': '%', 'log': False, 'bins': 20},
    'S':  {'color': '#fdcb6e', 'unit': '%', 'log': False, 'bins': 20},
}

# ================================================================
# 计算统计量（含剔除离群值后的质量加权均值）
# 离群值判定：改为更宽松的 99% 分位数截断（仅剔除极个别极端高值，如 Cu=15.6%）
# ================================================================
stats_table = {}
for col, cfg in elements.items():
    simple_avg = df[col].mean()
    mass_wt    = (df[col_mass] * df[col]).sum() / M
    median_v   = df[col].median()
    
    # 采用 99% 分位数作为上限，下限不截断（因为品位最小为0，极小值对质量加权影响不大）
    fence_lo   = df[col].min()
    fence_hi   = df[col].quantile(0.99)

    # 剔除离群值后的子集
    mask_clean  = (df[col] >= fence_lo) & (df[col] <= fence_hi)
    df_clean    = df[mask_clean]
    n_outliers  = (~mask_clean).sum()
    if len(df_clean) > 0:
        M_clean      = df_clean[col_mass].sum()
        mass_wt_rob  = (df_clean[col_mass] * df_clean[col]).sum() / M_clean
    else:
        mass_wt_rob  = mass_wt

    stats_table[col] = {
        '简单平均':       simple_avg,
        '质量加权平均':   mass_wt,
        '质量加权(去极值)': mass_wt_rob,
        '中位数':         median_v,
        'fence_lo': fence_lo, 'fence_hi': fence_hi,
        'n_outliers': n_outliers,
    }
    print(f"{col}: 质量加权={mass_wt:.4f}%  去极值加权={mass_wt_rob:.4f}%  "
          f"({n_outliers}个离群值被剔除, 栅栏=[{fence_lo:.3f}, {fence_hi:.3f}])")

# ================================================================
# 图1: 3×2 品位分布大图（直方图 + 箱线图）
# ================================================================
fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35)

for i, (col, cfg) in enumerate(elements.items()):
    ax_hist = fig.add_subplot(gs[0, i])
    ax_box  = fig.add_subplot(gs[1, i])

    data       = df[col]
    s_avg      = stats_table[col]['简单平均']
    mw_avg     = stats_table[col]['质量加权平均']
    mw_rob     = stats_table[col]['质量加权(去极值)']
    median_v   = stats_table[col]['中位数']
    n_out      = stats_table[col]['n_outliers']

    # --- 直方图 ---
    if cfg['log']:
        log_data = np.log10(data.clip(lower=1e-4))
        ax_hist.hist(log_data, bins=cfg['bins'], weights=df[col_mass], color=cfg['color'],
                     edgecolor='k', linewidth=0.5, alpha=0.75, density=False)
        ax_hist.set_xlabel(f'log10({col} 品位)', fontsize=11)
        ax_hist.axvline(np.log10(s_avg),  color='navy',    ls='--', lw=1.8, label=f'简单均值 {s_avg:.4f}%')
        ax_hist.axvline(np.log10(mw_avg), color='red',     ls='-',  lw=2.0, label=f'质量加权（全体）{mw_avg:.4f}%')
        ax_hist.axvline(np.log10(mw_rob), color='darkorange', ls='-.', lw=2.0, label=f'质量加权（去极值）{mw_rob:.4f}%')
        ax_hist.axvline(np.log10(median_v), color='green', ls=':', lw=1.5, label=f'中位数 {median_v:.4f}%')
    else:
        ax_hist.hist(data, bins=cfg['bins'], weights=df[col_mass], color=cfg['color'],
                     edgecolor='k', linewidth=0.5, alpha=0.75, density=False)
        ax_hist.set_xlabel(f'{col} 品位 (%)', fontsize=11)
        ax_hist.axvline(s_avg,    color='navy',       ls='--', lw=1.8, label=f'简单均值 {s_avg:.3f}%')
        ax_hist.axvline(mw_avg,   color='red',        ls='-',  lw=2.0, label=f'质量加权（全体）{mw_avg:.3f}%')
        ax_hist.axvline(mw_rob,   color='darkorange', ls='-.', lw=2.0, label=f'质量加权（去极值）{mw_rob:.3f}%')
        ax_hist.axvline(median_v, color='green',      ls=':',  lw=1.5, label=f'中位数 {median_v:.3f}%')

    ax_hist.set_ylabel('质量分布 (g)', fontsize=11)
    ax_hist.set_title(f'{col} 品位质量分布', fontsize=12, fontweight='bold')
    
    # --- 累积质量曲线 (双 Y 轴) ---
    ax_hist2 = ax_hist.twinx()
    sort_idx = np.argsort(data.values)
    if cfg['log']:
        sorted_x = np.log10(data.values[sort_idx].clip(min=1e-4))
    else:
        sorted_x = data.values[sort_idx]
    cum_mass_pct = df[col_mass].values[sort_idx].cumsum() / M * 100
    ax_hist2.plot(sorted_x, cum_mass_pct, color='purple', lw=2, alpha=0.85)
    ax_hist2.set_ylabel('累积质量 (%)', color='purple', fontsize=11)
    ax_hist2.tick_params(axis='y', labelcolor='purple')
    ax_hist2.set_ylim(0, 105)

    # legend 显示去极值说明（改到左上角以免遮挡右上角的累积曲线）
    handles, labels = ax_hist.get_legend_handles_labels()
    # 也把累积曲线加进图例
    line_cum = plt.Line2D([0], [0], color='purple', lw=2, alpha=0.85, label='累积质量 %')
    handles.append(line_cum)
    labels.append('累积质量 %')
    ax_hist.legend(handles, labels, fontsize=7.5, loc='upper left',
                   title=f'去极值: 剔除{n_out}个>99%分位值', title_fontsize=7)


    # (已移除"质量加权比简单均值高"注释框，保持图面简洁)


    # --- 箱线图（按密度分段着色）---
    density_bins   = [0, 2.5, 2.7, 3.0, 3.2, 10.0]
    density_labels = ['<2.5', '2.5~2.7', '2.7~3.0', '3.0~3.2', '>3.2']
    seg_colors = ['#74b9ff', '#a29bfe', '#55efc4', '#fdcb6e', '#e17055']

    seg_data = []
    for lo, hi in zip(density_bins[:-1], density_bins[1:]):
        mask = (df[col_density] > lo) & (df[col_density] <= hi)
        seg_data.append(df.loc[mask, col].values)

    bp = ax_box.boxplot(seg_data, patch_artist=True, widths=0.5,
                        medianprops=dict(color='k', lw=2))
    for patch, color in zip(bp['boxes'], seg_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax_box.set_xticklabels(density_labels, fontsize=9)
    ax_box.set_xlabel('密度段 (g/ml)', fontsize=11)
    ax_box.set_ylabel(f'{col} 品位 (%)', fontsize=11)
    ax_box.set_title(f'{col} 品位按密度分段箱线图', fontsize=12, fontweight='bold')
    ax_box.grid(True, axis='y', alpha=0.3)
    # 同时画出全体质量加权均值（红实线）和去极值质量加权均值（橙点划线）
    ax_box.axhline(mw_avg, color='red',        ls='-',  lw=1.5, alpha=0.8, label=f'质量加权（全体）{mw_avg:.3f}%')
    ax_box.axhline(mw_rob, color='darkorange', ls='-.', lw=1.5, alpha=0.9, label=f'质量加权（去极值）{mw_rob:.3f}%')

    # Cu 箱线图：离群值 32531 (Cu=15.6%) 会压缩其余段的视觉细节
    # → 将 Y 轴上限截断至 95% 分位数的 1.3 倍，并用文字标注被截断的最大值
    if col == 'Cu':
        p95   = df[col].quantile(0.95)
        y_max = p95 * 1.3
        ax_box.set_ylim(bottom=0, top=y_max)
        clipped = df[df[col] > y_max][col]
        if len(clipped) > 0:
            ax_box.text(0.98, 0.97,
                        f'注: {len(clipped)} 个点超出坐标轴\n最大值 {clipped.max():.2f}%',
                        transform=ax_box.transAxes, va='top', ha='right',
                        fontsize=8, color='#c0392b',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#fadbd8', alpha=0.85))

    ax_box.legend(fontsize=8)

plt.suptitle('各元素品位分布：频率直方图 + 密度分段箱线图\n'
             '（红线=质量加权均值，蓝虚线=简单算术均值，绿虚线=中位数）',
             fontsize=13, fontweight='bold')
plt.savefig(os.path.join(OUTPUT_DIR, '09_grade_distributions.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("图已保存: 09_grade_distributions.png")

# ================================================================
# 图2: 质量 vs 品位 散点图（验证质量-品位相关性）
# ================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, (col, cfg) in enumerate(elements.items()):
    ax = axes[i]
    sc = ax.scatter(df[col], df[col_mass], c=df[col_density],
                    cmap='RdYlGn', s=50, alpha=0.75, edgecolors='k', lw=0.4)
    plt.colorbar(sc, ax=ax, label='密度 (g/ml)')
    corr = df[col].corr(df[col_mass])
    ax.set_xlabel(f'{col} 品位 (%)', fontsize=11)
    ax.set_ylabel('矿石质量 (g)', fontsize=11)
    ax.set_title(f'{col} 品位 vs 矿石质量\n(r={corr:.3f})', fontsize=12, fontweight='bold')
    ax.axhline(df[col_mass].mean(), color='gray', ls='--', lw=1, alpha=0.6, label='平均质量')
    ax.legend(fontsize=8)
plt.suptitle('矿石质量 vs 品位关系（颜色=密度）\n'
             '质量加权≠简单平均 是因为大块矿石品位偏高',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '10_mass_vs_grade.png'), dpi=150, bbox_inches='tight')
plt.close()
print("图已保存: 10_mass_vs_grade.png")
print("完成。")
