import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

def load_any_dual_pixels(file_path, flip=False):
    """Loads low and high energy pixels and detects if it's a step sample (list) or simple sample (ndarray)."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    low = data['pixels_low']
    high = data['pixels_high']
    is_step = isinstance(low, list)
    
    if is_step and flip:
        return low[::-1], high[::-1], True
    return low, high, is_step

def plot_step_means(configs, title_suffix, save_path):
    """Plots means for stepped samples (1-10 steps)."""
    plt.figure(figsize=(12, 7))
    steps = np.arange(1, 11)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for i, cfg in enumerate(configs):
        low, high = cfg['data']
        label = cfg['label']
        color = colors[i % 10]
        m_low = [np.mean(p) for p in low]
        m_high = [np.mean(p) for p in high]
        plt.plot(steps, m_low, 'o-', color=color, label=f'{label} - Low')
        plt.plot(steps, m_high, 's--', color=color, label=f'{label} - High', alpha=0.6)
    
    plt.title(f'Steps Mean Comparison: {title_suffix}', fontsize=14)
    plt.xlabel('Thickness Step (1-10)')
    plt.ylabel('Mean Intensity')
    plt.xticks(steps)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Step mean plot saved to {save_path}")

def plot_simple_correlation(configs_x, configs_y, label_x, label_y, title_suffix, save_path):
    """Plots a scatter correlation between two sets of simple samples (e.g. 125us vs 270us)."""
    plt.figure(figsize=(8, 8))
    
    # Extract means
    low_x = [np.mean(c['data'][0]) for c in configs_x]
    high_x = [np.mean(c['data'][1]) for c in configs_x]
    low_y = [np.mean(c['data'][0]) for c in configs_y]
    high_y = [np.mean(c['data'][1]) for c in configs_y]
    
    # Plot Scatter
    plt.scatter(low_x, low_y, marker='o', color='blue', s=60, label='Low Energy', edgecolors='white', alpha=0.9)
    plt.scatter(high_x, high_y, marker='s', color='orange', s=60, label='High Energy', edgecolors='white', alpha=0.9)
    
    # Add Identity Line (Reference)
    all_vals = low_x + high_x + low_y + high_y
    v_min, v_max = min(all_vals)*0.95, max(all_vals)*1.05
    plt.plot([v_min, v_max], [v_min, v_max], 'k--', alpha=0.5, label='Identity (y=x)')
    
    # Annotate points with labels
    for i, cfg in enumerate(configs_x):
        plt.annotate(cfg['label'], (low_x[i], low_y[i]), xytext=(5,5), textcoords='offset points', fontsize=9)
        plt.annotate(cfg['label'], (high_x[i], high_y[i]), xytext=(5,5), textcoords='offset points', fontsize=9)

    plt.title(f'Intensity Correlation: {title_suffix}', fontsize=14)
    plt.xlabel(f'{label_x} Mean Intensity')
    plt.ylabel(f'{label_y} Mean Intensity')
    plt.axis('equal')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Correlation plot saved to {save_path}")

def plot_simple_means(configs, title_suffix, save_path):
    """Plots means for simple samples as a comparative bar chart."""
    plt.figure(figsize=(10, 6))
    labels = [c['label'] for c in configs]
    m_lows = [np.mean(c['data'][0]) for c in configs]
    m_highs = [np.mean(c['data'][1]) for c in configs]
    
    x = np.arange(len(labels))
    width = 0.35
    plt.bar(x - width/2, m_lows, width, label='Low Energy', color='skyblue')
    plt.bar(x + width/2, m_highs, width, label='High Energy', color='coral')
    
    plt.title(f'Simple Sample Means: {title_suffix}', fontsize=14)
    plt.xticks(x, labels, rotation=45)
    plt.ylabel('Mean Intensity')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Simple mean plot saved to {save_path}")

def plot_step_hist_grid(configs, channel_name, title_suffix, save_path):
    """Plots density histograms for stepped datasets in a 2x5 grid."""
    fig, axes = plt.subplots(2, 5, figsize=(22, 10), constrained_layout=True)
    axes = axes.flatten()
    colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))
    
    idx = 0 if channel_name.lower() == 'low' else 1
    for i in range(10):
        ax = axes[i]
        all_d = [cfg['data'][idx][i] for cfg in configs]
        v_min = min(np.min(d) for d in all_d)
        v_max = max(np.max(d) for d in all_d)
        bins = np.linspace(v_min, v_max, 50)
        
        for j, cfg in enumerate(configs):
            ax.hist(all_d[j], bins=bins, alpha=0.4, label=cfg['label'], color=colors[j], density=True)
        
        ax.set_title(f'Step {i+1}')
        ax.set_xlabel('Intensity')
        if i == 0:
            ax.legend(loc='upper right', prop={'size': 9})
    
    fig.suptitle(f'Steps ({title_suffix}): {channel_name} Energy', fontsize=18)
    plt.savefig(save_path)
    print(f"Step histogram grid saved to {save_path}")

def plot_simple_hist_grid(configs_x, configs_y, channel_name, title_suffix, save_path):
    """Plots a grid of histograms where each subplot compares a specific ore from Dataset X vs Dataset Y."""
    num_ores = len(configs_x)
    cols = 2
    rows = (num_ores + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 5 * rows), squeeze=False)
    axes = axes.flatten()
    
    idx = 0 if channel_name.lower() == 'low' else 1
    
    for i in range(num_ores):
        ax = axes[i]
        d_x = configs_x[i]['data'][idx]
        d_y = configs_y[i]['data'][idx]
        l_x = configs_x[i]['label']
        l_y = configs_y[i]['label']
        
        v_min, v_max = min(np.min(d_x), np.min(d_y)), max(np.max(d_x), np.max(d_y))
        bins = np.linspace(v_min, v_max, 60)
        
        ax.hist(d_x, bins=bins, alpha=0.4, label=f"125us-{l_x}", color='blue', density=True)
        ax.hist(d_y, bins=bins, alpha=0.4, label=f"270us-{l_y}", color='orange', density=True)
        
        ax.set_title(f'Comparison: {l_x} vs {l_y}')
        ax.set_xlabel('Intensity')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    # Hide unused axes
    for i in range(num_ores, len(axes)):
        axes[i].axis('off')
        
    fig.suptitle(f'Simple Sample ({title_suffix}): {channel_name} Energy Per-Ore Histograms', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    print(f"Simple histogram grid saved to {save_path}")

def plot_simple_hist(configs, channel_name, title_suffix, save_path):
    # (Kept for non-XY mode simple comparisons if any)
    plt.figure(figsize=(12, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))
    idx = 0 if channel_name.lower() == 'low' else 1
    all_d = [cfg['data'][idx] for cfg in configs]
    v_min, v_max = min(np.min(d) for d in all_d), max(np.max(d) for d in all_d)
    bins = np.linspace(v_min, v_max, 80)
    
    for j, cfg in enumerate(configs):
        plt.hist(all_d[j], bins=bins, alpha=0.3, label=cfg['label'], color=colors[j], density=True)
    
    plt.title(f'Histogram ({title_suffix}): {channel_name} Energy', fontsize=14)
    plt.xlabel('Intensity'); plt.ylabel('Density'); plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(save_path); print(f"Simple overall histogram saved to {save_path}")

def run_comparison(configs_desc, title_suffix, prefix, is_xy=False):
    """Main entry point for routing to different plotting modes."""
    os.makedirs('results/Tube_comparison', exist_ok=True)
    
    if is_xy and len(configs_desc) == 2:
        # Correlation Mode (Group X vs Group Y)
        desc_x, desc_y = configs_desc[0], configs_desc[1]
        configs_x = []
        for cd in desc_x:
            low, high, _ = load_any_dual_pixels(cd['path'], flip=cd.get('flip', False))
            configs_x.append({'data': (low, high), 'label': cd['label']})
        configs_y = []
        for cd in desc_y:
            low, high, _ = load_any_dual_pixels(cd['path'], flip=cd.get('flip', False))
            configs_y.append({'data': (low, high), 'label': cd['label']})
            
        # 1. Scatter Correlation
        plot_simple_correlation(configs_x, configs_y, "125us", "270us", title_suffix, f'results/Tube_comparison/{prefix}_correlation.png')
        
        # 2. Per-Ore Histogram Grid Comparison
        plot_simple_hist_grid(configs_x, configs_y, "Low", title_suffix, f'results/Tube_comparison/{prefix}_hist_low.png')
        plot_simple_hist_grid(configs_x, configs_y, "High", title_suffix, f'results/Tube_comparison/{prefix}_hist_high.png')
    else:
        # Standard Mode (Sequential list)
        configs = []
        is_step_run = False
        for i, cd in enumerate(configs_desc):
            low, high, is_step = load_any_dual_pixels(cd['path'], flip=cd.get('flip', False))
            if i == 0: is_step_run = is_step
            configs.append({'data': (low, high), 'label': cd['label']})
        
        if is_step_run:
            plot_step_means(configs, title_suffix, f'results/Tube_comparison/{prefix}_means.png')
            plot_step_hist_grid(configs, "Low", title_suffix, f'results/Tube_comparison/{prefix}_hist_low.png')
            plot_step_hist_grid(configs, "High", title_suffix, f'results/Tube_comparison/{prefix}_hist_high.png')
        else:
            plot_simple_means(configs, title_suffix, f'results/Tube_comparison/{prefix}_means.png')
            plot_simple_hist(configs, "Low", title_suffix, f'results/Tube_comparison/{prefix}_hist_low.png')
            plot_simple_hist(configs, "High", title_suffix, f'results/Tube_comparison/{prefix}_hist_high.png')

def main():
    # Setup plotting aesthetics for Chinese text
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False

    # 1. Comparison of Steps (125us vs 270us) - Stepped Mode
    print("\n=== RUNNING STEP COMPARISON (125us vs 270us) ===")
    configs_step = [
        {"path": r'results/TYM_test/pixel_values/160kv-2mA-125us-0.5pF-disc-post_calib_step_sample_9_data.pkl', "label": "125us"},
        {"path": r'results/TYM_test/pixel_values/160kv-2mA-270us-0.5pF-disc-post_calib_step_sample_9_data.pkl', "label": "270us"}
    ]
    run_comparison(configs_step, "Steps: Exposure Time", "TYM_Exposure_Steps")
    
    # 2. Comparison of Ores (125us vs 270us) - Correlation Mode & Per-Ore Hist Grid
    print("\n=== RUNNING ORE CORRELATION & HIST GRID (125us vs 270us) ===")
    
    # MANUAL INDICES SPECIFICATION
    indices_125 = [2, 3, 0] # Ore numbers extracted from 125us data
    indices_270 = [2, 3, 4] # Ore numbers extracted from 270us data (can match differently)
    
    configs_x = [{"path": f'results/TYM_test/pixel_values/160kv-2mA-125us-0.5pF-ore-post_calib_ore_{i}_data.pkl', "label": f"Ore{i}"} for i in indices_125]
    configs_y = [{"path": f'results/TYM_test/pixel_values/160kv-2mA-270us-0.5pF-ore&step-post_calib_ore_{j}_data.pkl', "label": f"Ore{j}"} for j in indices_270]
    
    run_comparison([configs_x, configs_y], "Ores: 125us vs 270us", "TYM_Exposure_Ores", is_xy=True)

if __name__ == "__main__":
    main()
