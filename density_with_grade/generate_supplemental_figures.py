"""
补充合并版图像：
  - 修正 01c 相关热力图列顺序（与原版 01 保持一致）
  - 生成 09c（合并批次品位分布，Cu+Fe 两元素）
  - 生成 10c（合并批次质量 vs 品位散点）
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
OUTPUT_DIR = 'analysis_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 载入合并数据（与 generate_combined_figures.py 一致）
# ============================================================
df_325 = pd.read_excel('工作簿1.xlsx', sheet_name='0325')
df_325.columns = ['矿石编号', 'Cu', 'Fe', 'S', '质量', '密度']
df_325['批次'] = '0325'

df_520_raw = pd.read_excel('工作簿1.xlsx', sheet_name='Sheet1')
df_520_raw.columns = ['序号', '质量', '矿石编号', 'Fe', 'Cu', 'S', 'Volume_cm3', 'density']
df_520 = df_520_raw[['矿石编号', 'Cu', 'Fe', '质量', 'density']].copy()
df_520 = df_520.rename(columns={'density': '密度'})
df_520['密度'] = pd.to_numeric(df_520['密度'], errors='coerce')
df_520_clean = df_520[(df_520['密度'] >= 1.0) & (df_520['密度'] <= 6.0)].copy()
df_520_clean['S'] = np.nan
df_520_clean['批次'] = '0520'

df = pd.concat([df_325, df_520_clean[['矿石编号','Cu','Fe','S','质量','密度','批次']]],
               ignore_index=True)

M = df['质量'].sum()
colors_batch = {'0325': '#e17055', '0520': '#74b9ff'}

# ============================================================
# 图 01c（修正版）: 相关系数热力图 — 列顺序与原版一致
# 原版顺序: 密度, Cu, Fe, S, 总金属品位
# ============================================================
df['总金属品位'] = df['Cu'] + df['Fe'] + df['S']
corr_cols = ['密度', 'Cu', 'Fe', 'S', '总金属品位']
df_corr = df[corr_cols].copy()
corr_matrix = df_corr.corr()

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
            vmin=-1, vmax=1, center=0, linewidths=0.5, ax=ax,
            xticklabels=corr_cols, yticklabels=corr_cols)
ax.set_title('相关系数热力图\n（合并批次，密度/Cu/Fe/S/总金属品位，顺序与 0325 版一致）',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '01c_correlation_heatmap_combined.png'), dpi=150)
plt.close()
print('01c 修正完成')

# ============================================================
# 图 09c: 品位分布大图（合并批次，Cu + Fe 两元素，两批次着色）
# ============================================================
elements = {
    'Cu': {'color': '#e17055', 'log': True,  'bins': 30},
    'Fe': {'color': '#74b9ff', 'log': False, 'bins': 20},
}

density_bins   = [0, 2.5, 2.7, 3.0, 3.2, 10.0]
density_labels = ['<2.5', '2.5~2.7', '2.7~3.0', '3.0~3.2', '>3.2']
seg_colors = ['#74b9ff', '#a29bfe', '#55efc4', '#fdcb6e', '#e17055']

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)

for i, (col, cfg) in enumerate(elements.items()):
    ax_hist = fig.add_subplot(gs[0, i])
    ax_box  = fig.add_subplot(gs[1, i])

    data_all = df[col]
    mass_all = df['质量']
    M_all = mass_all.sum()

    # 统计量
    s_avg  = data_all.mean()
    mw_avg = (mass_all * data_all).sum() / M_all
    p99    = data_all.quantile(0.99)
    mask_clean = data_all <= p99
    df_c = df[mask_clean]
    mw_rob = (df_c['质量'] * df_c[col]).sum() / df_c['质量'].sum()
    n_out  = (~mask_clean).sum()
    median_v = data_all.median()

    # --- 直方图：两批次分层叠加 ---
    for bat, bc in colors_batch.items():
        sub = df[df['批次'] == bat]
        vals = sub[col]
        wts  = sub['质量']
        if cfg['log']:
            ax_hist.hist(np.log10(vals.clip(lower=1e-4)), bins=cfg['bins'],
                         weights=wts, color=bc, edgecolor='k', lw=0.4,
                         alpha=0.5, label=bat)
        else:
            ax_hist.hist(vals, bins=cfg['bins'], weights=wts, color=bc,
                         edgecolor='k', lw=0.4, alpha=0.5, label=bat)

    # 均值线
    if cfg['log']:
        ax_hist.axvline(np.log10(s_avg),  color='navy',       ls='--', lw=1.8, label=f'简单均值 {s_avg:.4f}%')
        ax_hist.axvline(np.log10(mw_avg), color='red',        ls='-',  lw=2.0, label=f'质量加权（全体）{mw_avg:.4f}%')
        ax_hist.axvline(np.log10(mw_rob), color='darkorange', ls='-.', lw=2.0, label=f'质量加权（去极值）{mw_rob:.4f}%')
        ax_hist.axvline(np.log10(median_v), color='green',    ls=':',  lw=1.5, label=f'中位数 {median_v:.4f}%')
        ax_hist.set_xlabel(f'log10({col} 品位)', fontsize=11)
    else:
        ax_hist.axvline(s_avg,    color='navy',       ls='--', lw=1.8, label=f'简单均值 {s_avg:.3f}%')
        ax_hist.axvline(mw_avg,   color='red',        ls='-',  lw=2.0, label=f'质量加权（全体）{mw_avg:.3f}%')
        ax_hist.axvline(mw_rob,   color='darkorange', ls='-.', lw=2.0, label=f'质量加权（去极值）{mw_rob:.3f}%')
        ax_hist.axvline(median_v, color='green',      ls=':',  lw=1.5, label=f'中位数 {median_v:.3f}%')
        ax_hist.set_xlabel(f'{col} 品位 (%)', fontsize=11)

    # 累积质量曲线
    ax_h2 = ax_hist.twinx()
    sort_idx = np.argsort(data_all.values)
    sorted_x = (np.log10(data_all.values[sort_idx].clip(min=1e-4))
                 if cfg['log'] else data_all.values[sort_idx])
    cum_pct = mass_all.values[sort_idx].cumsum() / M_all * 100
    ax_h2.plot(sorted_x, cum_pct, color='purple', lw=2, alpha=0.85, label='累积质量 %')
    ax_h2.set_ylabel('累积质量 (%)', color='purple', fontsize=11)
    ax_h2.tick_params(axis='y', labelcolor='purple')
    ax_h2.set_ylim(0, 105)

    ax_hist.set_ylabel('质量分布 (g)', fontsize=11)
    ax_hist.set_title(f'{col} 品位质量分布（合并批次）', fontsize=12, fontweight='bold')
    handles, labels = ax_hist.get_legend_handles_labels()
    line_cum = plt.Line2D([0],[0], color='purple', lw=2, alpha=0.85, label='累积质量 %')
    handles.append(line_cum); labels.append('累积质量 %')
    ax_hist.legend(handles, labels, fontsize=7, loc='upper left',
                   title=f'去极值: 剔除{n_out}个>99%', title_fontsize=7)

    # --- 箱线图（密度分段，两批次合并）---
    seg_data = []
    for lo, hi in zip(density_bins[:-1], density_bins[1:]):
        mask = (df['密度'] > lo) & (df['密度'] <= hi)
        seg_data.append(df.loc[mask, col].values)

    bp = ax_box.boxplot(seg_data, patch_artist=True, widths=0.5,
                        medianprops=dict(color='k', lw=2))
    for patch, color in zip(bp['boxes'], seg_colors):
        patch.set_facecolor(color); patch.set_alpha(0.8)
    ax_box.set_xticklabels(density_labels, fontsize=9)
    ax_box.set_xlabel('密度段 (g/ml)', fontsize=11)
    ax_box.set_ylabel(f'{col} 品位 (%)', fontsize=11)
    ax_box.set_title(f'{col} 品位按密度分段箱线图（合并）', fontsize=12, fontweight='bold')
    ax_box.grid(True, axis='y', alpha=0.3)
    ax_box.axhline(mw_avg, color='red',        ls='-',  lw=1.5, alpha=0.8,
                   label=f'质量加权（全体）{mw_avg:.3f}%')
    ax_box.axhline(mw_rob, color='darkorange', ls='-.', lw=1.5, alpha=0.9,
                   label=f'质量加权（去极值）{mw_rob:.3f}%')

    if col == 'Cu':
        p95 = data_all.quantile(0.95)
        y_max = p95 * 1.3
        ax_box.set_ylim(bottom=0, top=y_max)
        clipped = data_all[data_all > y_max]
        if len(clipped) > 0:
            ax_box.text(0.98, 0.97,
                        f'注: {len(clipped)} 个点超出坐标轴\n最大值 {clipped.max():.2f}%',
                        transform=ax_box.transAxes, va='top', ha='right',
                        fontsize=8, color='#c0392b',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#fadbd8', alpha=0.85))
    ax_box.legend(fontsize=8)

plt.suptitle('合并批次（0325+0520）Cu / Fe 品位分布：质量直方图 + 密度分段箱线图\n'
             '（红线=质量加权均值，蓝虚线=简单算术均值，绿虚线=中位数；两批次颜色叠加）',
             fontsize=12, fontweight='bold')
plt.savefig(os.path.join(OUTPUT_DIR, '09c_grade_distributions_combined.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print('09c 完成')

# ============================================================
# 图 10c: 质量 vs 品位 散点（合并批次，两批次着色）
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for i, (col, cfg) in enumerate(elements.items()):
    ax = axes[i]
    for bat, bc in colors_batch.items():
        sub = df[df['批次'] == bat]
        sc = ax.scatter(sub[col], sub['质量'], c=sub['密度'],
                        cmap='RdYlGn', s=55, alpha=0.75,
                        edgecolors=bc, lw=1.0,
                        label=f'{bat}', vmin=1.8, vmax=4.6)
    plt.colorbar(sc, ax=ax, label='密度 (g/ml)')
    corr_all = df[col].corr(df['质量'])
    ax.set_xlabel(f'{col} 品位 (%)', fontsize=11)
    ax.set_ylabel('矿石质量 (g)', fontsize=11)
    ax.set_title(f'{col} 品位 vs 矿石质量\n(全体 r={corr_all:.3f})', fontsize=12, fontweight='bold')
    ax.axhline(df['质量'].mean(), color='gray', ls='--', lw=1, alpha=0.6, label='平均质量')
    ax.legend(fontsize=8)

plt.suptitle('合并批次（0325+0520）矿石质量 vs 品位（颜色=密度，外框颜色=批次）',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '10c_mass_vs_grade_combined.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print('10c 完成')
print('\n全部完成。')
