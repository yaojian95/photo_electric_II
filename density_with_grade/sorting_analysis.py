"""
基于密度阈值的矿石分选指标分析
================================
分选规则: 密度 > 阈值 ρ_t → 进入精矿 (concentrate)
          密度 ≤ 阈值 ρ_t → 丢弃为尾矿 (tailings)

分选指标定义:
    产率   γ    = M_conc / M_feed × 100%             （精矿质量占给矿质量比）
    品位   β    = Σ(m_i × w_i) / M_conc              （精矿中各元素的质量加权品位）
    回收率 ε    = (M_conc × β) / (M_feed × α) × 100% （进入精矿的元素质量比）
    富集比 k    = β / α                               （精矿品位 / 给矿品位）
    尾矿品位 β_t = (1 - ε/100 × α × M_feed/M_feed) ... 用质量守恒计算
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = 'analysis_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===========================================================
# 加载数据
# ===========================================================
df = pd.read_excel('工作簿1.xlsx')
col_id      = '矿石编号'
col_cu      = 'Cu'
col_fe      = 'Fe'
col_s       = 'S'
col_mass    = '质量'
col_density = '密度(g/ml)'

# 给矿（全体）总量与平均品位
M_feed    = df[col_mass].sum()
alpha_cu  = (df[col_mass] * df[col_cu]).sum() / M_feed  # 给矿Cu品位
alpha_fe  = (df[col_mass] * df[col_fe]).sum() / M_feed  # 给矿Fe品位
alpha_s   = (df[col_mass] * df[col_s ]).sum() / M_feed  # 给矿S品位

print("=== 给矿指标 ===")
print(f"  总质量  : {M_feed:.1f} g")
print(f"  Cu 品位 : {alpha_cu:.4f} %")
print(f"  Fe 品位 : {alpha_fe:.3f} %")
print(f"  S  品位 : {alpha_s:.3f} %")

# ===========================================================
# 全局密度阈值扫描
# ===========================================================
# 使用排序后每个唯一密度值作为阈值候选，步长加密
rho_min = df[col_density].min()
rho_max = df[col_density].max()
thresholds = np.linspace(rho_min, rho_max, 500)

records = []
for rho_t in thresholds:
    conc = df[df[col_density] > rho_t]
    tail = df[df[col_density] <= rho_t]

    if len(conc) == 0:
        # 没有精矿
        records.append({
            'rho_t': rho_t, 'n_conc': 0,
            'yield_pct': 0.0,
            'beta_cu': np.nan, 'beta_fe': np.nan, 'beta_s': np.nan,
            'rec_cu': 0.0, 'rec_fe': 0.0, 'rec_s': 0.0,
            'enrich_cu': np.nan, 'enrich_fe': np.nan, 'enrich_s': np.nan,
            'tail_cu': alpha_cu, 'tail_fe': alpha_fe, 'tail_s': alpha_s,
        })
        continue

    M_conc = conc[col_mass].sum()
    M_tail = tail[col_mass].sum()

    beta_cu = (conc[col_mass] * conc[col_cu]).sum() / M_conc
    beta_fe = (conc[col_mass] * conc[col_fe]).sum() / M_conc
    beta_s  = (conc[col_mass] * conc[col_s ]).sum() / M_conc

    rec_cu  = M_conc * beta_cu  / (M_feed * alpha_cu)  * 100
    rec_fe  = M_conc * beta_fe  / (M_feed * alpha_fe)  * 100
    rec_s   = M_conc * beta_s   / (M_feed * alpha_s )  * 100

    yield_pct = M_conc / M_feed * 100

    # 尾矿品位（质量守恒）
    tail_cu = (M_feed * alpha_cu - M_conc * beta_cu) / M_tail if M_tail > 0 else np.nan
    tail_fe = (M_feed * alpha_fe - M_conc * beta_fe) / M_tail if M_tail > 0 else np.nan
    tail_s  = (M_feed * alpha_s  - M_conc * beta_s ) / M_tail if M_tail > 0 else np.nan

    records.append({
        'rho_t': rho_t, 'n_conc': len(conc),
        'yield_pct': yield_pct,
        'beta_cu': beta_cu, 'beta_fe': beta_fe, 'beta_s': beta_s,
        'rec_cu': rec_cu, 'rec_fe': rec_fe, 'rec_s': rec_s,
        'enrich_cu': beta_cu / alpha_cu, 'enrich_fe': beta_fe / alpha_fe, 'enrich_s': beta_s / alpha_s,
        'tail_cu': tail_cu, 'tail_fe': tail_fe, 'tail_s': tail_s,
    })

res = pd.DataFrame(records)

# ===========================================================
# 可视化 1: 三元素回收率 vs 产率（选别曲线）
# ===========================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.plot(res['yield_pct'], res['rec_cu'], color='#e17055', lw=2.0, label='Cu 回收率')
ax.plot(res['yield_pct'], res['rec_fe'], color='#74b9ff', lw=2.0, label='Fe 回收率')
ax.plot(res['yield_pct'], res['rec_s'],  color='#fdcb6e', lw=2.0, label='S 回收率')
ax.plot(res['yield_pct'], res['yield_pct'], 'k--', lw=1.2, alpha=0.5, label='γ=ε 参考线（无富集）')
ax.set_xlabel('产率 γ (%)', fontsize=12)
ax.set_ylabel('回收率 ε (%)', fontsize=12)
ax.set_title('选别曲线：产率 vs 各元素回收率\n（密度阈值扫描，密度>阈值入精矿）', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.set_xlim(0, 100); ax.set_ylim(0, 105)
ax.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(res['yield_pct'], res['enrich_cu'], color='#e17055', lw=2.0, label='Cu 富集比')
ax2.plot(res['yield_pct'], res['enrich_fe'], color='#74b9ff', lw=2.0, label='Fe 富集比')
ax2.plot(res['yield_pct'], res['enrich_s'],  color='#fdcb6e', lw=2.0, label='S 富集比')
ax2.axhline(1.0, color='k', ls='--', lw=1.2, alpha=0.5, label='富集比=1（无富集）')
ax2.set_xlabel('产率 γ (%)', fontsize=12)
ax2.set_ylabel('富集比 k', fontsize=12)
ax2.set_title('富集比 vs 产率\n（越低产率，富集比越高）', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.set_xlim(0, 100)
ax2.set_ylim(0, max(res['enrich_cu'].replace([np.inf], np.nan).dropna().max() * 1.1, 5))
ax2.grid(True, alpha=0.3)

plt.suptitle('密度阈值分选：选别指标全局扫描', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '06_sorting_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print("图1 已保存: 06_sorting_curves.png")

# ===========================================================
# 可视化 2: 各指标 vs 密度阈值
# ===========================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
ax.plot(res['rho_t'], res['yield_pct'], color='steelblue', lw=2)
ax.set_xlabel('密度阈值 ρ_t (g/ml)', fontsize=11)
ax.set_ylabel('产率 γ (%)', fontsize=11)
ax.set_title('产率 vs 密度阈值', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(res['rho_t'], res['rec_cu'], color='#e17055', lw=2, label='Cu')
ax.plot(res['rho_t'], res['rec_fe'], color='#74b9ff', lw=2, label='Fe')
ax.plot(res['rho_t'], res['rec_s'],  color='#fdcb6e', lw=2, label='S')
ax.set_xlabel('密度阈值 ρ_t (g/ml)', fontsize=11)
ax.set_ylabel('回收率 ε (%)', fontsize=11)
ax.set_title('各元素回收率 vs 密度阈值', fontsize=12, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.plot(res['rho_t'], res['beta_cu'], color='#e17055', lw=2, label='精矿 Cu 品位')
ax.axhline(alpha_cu, color='#e17055', ls='--', lw=1.2, alpha=0.6, label=f'给矿 Cu={alpha_cu:.4f}%')
ax.set_xlabel('密度阈值 ρ_t (g/ml)', fontsize=11)
ax.set_ylabel('精矿 Cu 品位 (%)', fontsize=11)
ax.set_title('精矿 Cu 品位 vs 密度阈值', fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.plot(res['rho_t'], res['enrich_cu'].replace([np.inf], np.nan), color='#e17055', lw=2, label='Cu 富集比')
ax.plot(res['rho_t'], res['enrich_fe'].replace([np.inf], np.nan), color='#74b9ff', lw=2, label='Fe 富集比')
ax.plot(res['rho_t'], res['enrich_s'].replace([np.inf], np.nan),  color='#fdcb6e', lw=2, label='S 富集比')
ax.axhline(1.0, color='k', ls='--', lw=1.2, alpha=0.5)
ax.set_xlabel('密度阈值 ρ_t (g/ml)', fontsize=11)
ax.set_ylabel('富集比 k', fontsize=11)
ax.set_title('各元素富集比 vs 密度阈值', fontsize=12, fontweight='bold')
ax.set_ylim(0, 12); ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

plt.suptitle('密度阈值分选指标随阈值变化曲线', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '07_sorting_metrics_vs_threshold.png'), dpi=150)
plt.close()
print("图2 已保存: 07_sorting_metrics_vs_threshold.png")

# ===========================================================
# 找寻典型阈值并输出详表
# ===========================================================
# 典型阈值选取策略:
#   T1: 密度 > 2.5  — 丢弃轻质脉石（宽松阈值，高回收）
#   T2: 密度 > 2.7  — 中等阈值（Q25附近）
#   T3: 密度 > 3.0  — 中等严格阈值
#   T4: 密度 > 3.2  — 严格阈值（富集精矿）
#   T_opt: Cu回收率/产率比最大点（最优Cu富集效率）

def get_metrics_at(rho_t):
    """
    计算给定密度阈值下的选矿指标。

    参数:
        rho_t (float): 密度阈值 (g/ml)，密度 > rho_t 的样本进入精矿。

    返回:
        dict: 包含精矿样本数、产率、精矿各元素品位、回收率、富集比和尾矿品位的字典。
    """
    conc = df[df[col_density] > rho_t]
    tail = df[df[col_density] <= rho_t]

    if len(conc) == 0:
        return None

    M_conc = conc[col_mass].sum()
    M_tail = tail[col_mass].sum() if len(tail) > 0 else 0

    beta_cu = (conc[col_mass] * conc[col_cu]).sum() / M_conc
    beta_fe = (conc[col_mass] * conc[col_fe]).sum() / M_conc
    beta_s  = (conc[col_mass] * conc[col_s ]).sum() / M_conc

    rec_cu  = M_conc * beta_cu / (M_feed * alpha_cu) * 100
    rec_fe  = M_conc * beta_fe / (M_feed * alpha_fe) * 100
    rec_s   = M_conc * beta_s  / (M_feed * alpha_s)  * 100

    tail_cu = (M_feed * alpha_cu - M_conc * beta_cu) / M_tail if M_tail > 0 else 0
    tail_fe = (M_feed * alpha_fe - M_conc * beta_fe) / M_tail if M_tail > 0 else 0
    tail_s  = (M_feed * alpha_s  - M_conc * beta_s ) / M_tail if M_tail > 0 else 0

    return {
        '阈值(g/ml)': rho_t,
        '精矿块数': len(conc),
        '产率γ(%)': round(M_conc / M_feed * 100, 2),
        'Cu精矿品位(%)': round(beta_cu, 4),
        'Fe精矿品位(%)': round(beta_fe, 3),
        'S精矿品位(%)':  round(beta_s,  3),
        'Cu回收率(%)':   round(rec_cu,  2),
        'Fe回收率(%)':   round(rec_fe,  2),
        'S回收率(%)':    round(rec_s,   2),
        'Cu富集比':      round(beta_cu / alpha_cu, 2),
        'Fe富集比':      round(beta_fe / alpha_fe, 2),
        'S富集比':       round(beta_s  / alpha_s,  2),
        'Cu尾矿品位(%)': round(tail_cu, 4),
    }

# 自动找"Cu回收率/产率"最大的阈值（即Cu富集效率最优点）
cu_efficiency = res['rec_cu'] / (res['yield_pct'] + 1e-9)  # 避免除以零
best_idx = cu_efficiency.idxmax()
rho_opt = res.loc[best_idx, 'rho_t']

typical_thresholds = [
    ('T1: 宽松（低损失）', 2.5),
    ('T2: 中等偏宽',       2.7),
    ('T3: 中等偏严',       3.0),
    ('T4: 严格（高富集）', 3.2),
    (f'T_opt: 最优Cu效率 ρ={rho_opt:.3f}', rho_opt),
]

print(f"\n=== 典型阈值分选指标对比 ===")
print(f"给矿 Cu品位={alpha_cu:.4f}%  Fe品位={alpha_fe:.3f}%  S品位={alpha_s:.3f}%\n")

rows = []
for name, rho_t in typical_thresholds:
    m = get_metrics_at(rho_t)
    if m:
        m['方案'] = name
        rows.append(m)
        print(f"[{name}] ρ_t={rho_t:.3f}")
        print(f"  精矿块数={m['精矿块数']}  产率={m['产率γ(%)']:.2f}%")
        print(f"  精矿品位: Cu={m['Cu精矿品位(%)']:.4f}%  Fe={m['Fe精矿品位(%)']:.3f}%  S={m['S精矿品位(%)']:.3f}%")
        print(f"  回收率:   Cu={m['Cu回收率(%)']:.1f}%  Fe={m['Fe回收率(%)']:.1f}%  S={m['S回收率(%)']:.1f}%")
        print(f"  富集比:   Cu={m['Cu富集比']:.2f}x  Fe={m['Fe富集比']:.2f}x  S={m['S富集比']:.2f}x")
        print(f"  尾矿Cu品位: {m['Cu尾矿品位(%)']:.4f}%")
        print()

df_result = pd.DataFrame(rows)[['方案', '阈值(g/ml)', '精矿块数', '产率γ(%)',
                                  'Cu精矿品位(%)', 'Fe精矿品位(%)', 'S精矿品位(%)',
                                  'Cu回收率(%)', 'Fe回收率(%)', 'S回收率(%)',
                                  'Cu富集比', 'Fe富集比', 'S富集比', 'Cu尾矿品位(%)']]
df_result.to_csv(os.path.join(OUTPUT_DIR, 'sorting_threshold_comparison.csv'),
                 index=False, encoding='utf-8-sig')
print("分选结果对比表已保存: sorting_threshold_comparison.csv")

# ===========================================================
# 可视化 3: 典型阈值对比柱状图
# ===========================================================
names_short = ['T1\nρ>2.5', 'T2\nρ>2.7', 'T3\nρ>3.0', 'T4\nρ>3.2', f'T_opt\nρ>{rho_opt:.2f}']
vals_yield  = [r['产率γ(%)']    for r in rows]
vals_rec_cu = [r['Cu回收率(%)'] for r in rows]
vals_rec_fe = [r['Fe回收率(%)'] for r in rows]
vals_enr_cu = [r['Cu富集比']    for r in rows]
vals_beta_cu= [r['Cu精矿品位(%)'] for r in rows]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
x = np.arange(len(names_short))
w = 0.28

ax = axes[0]
b1 = ax.bar(x - w/2, vals_yield,  width=w, label='产率 γ',    color='steelblue', edgecolor='k', lw=0.7)
b2 = ax.bar(x + w/2, vals_rec_cu, width=w, label='Cu 回收率', color='#e17055',   edgecolor='k', lw=0.7)
ax.set_xticks(x); ax.set_xticklabels(names_short, fontsize=9)
ax.set_ylabel('%', fontsize=12)
ax.set_title('产率 vs Cu 回收率', fontsize=12, fontweight='bold')
ax.set_ylim(0, 115)
ax.legend(fontsize=9)
for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)

ax = axes[1]
bars = ax.bar(x, vals_enr_cu, color='#e17055', edgecolor='k', lw=0.7)
ax.axhline(1.0, color='k', ls='--', lw=1.2, alpha=0.5, label='无富集基准线')
ax.set_xticks(x); ax.set_xticklabels(names_short, fontsize=9)
ax.set_ylabel('富集比 k', fontsize=12)
ax.set_title('Cu 富集比', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
for bar, v in zip(bars, vals_enr_cu):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{v:.1f}x', ha='center', va='bottom', fontsize=9)

ax = axes[2]
bars = ax.bar(x, vals_beta_cu, color='#a29bfe', edgecolor='k', lw=0.7)
ax.axhline(alpha_cu, color='k', ls='--', lw=1.2, alpha=0.5, label=f'给矿品位 {alpha_cu:.4f}%')
ax.set_xticks(x); ax.set_xticklabels(names_short, fontsize=9)
ax.set_ylabel('精矿 Cu 品位 (%)', fontsize=12)
ax.set_title('精矿 Cu 品位', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
for bar, v in zip(bars, vals_beta_cu):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
            f'{v:.3f}%', ha='center', va='bottom', fontsize=8)

plt.suptitle('典型密度阈值分选方案对比', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '08_threshold_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("图3 已保存: 08_threshold_comparison.png")
print("\n全部分析完成。")
