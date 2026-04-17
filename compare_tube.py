import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

def load_dual_step_pixels(file_path, flip=False):
    """Loads low and high energy pixels, with optional flipping for thickness alignment."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    low = data['pixels_low']
    high = data['pixels_high']
    if not isinstance(low, list):
        raise ValueError(f"File {file_path} does not contain a list of steps.")
    
    if flip:
        return low[::-1], high[::-1]
    return low, high

def plot_adaptive_means(configs, title_suffix, save_path):
    """Plots means for a variable number of datasets (Low as solid, High as dashed)."""
    plt.figure(figsize=(12, 7))
    steps = np.arange(1, 11)
    
    # Use a standard color cycle
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for i, cfg in enumerate(configs):
        low, high = cfg['data']
        label = cfg['label']
        color = colors[i % 10]
        
        m_low = [np.mean(p) for p in low]
        m_high = [np.mean(p) for p in high]
        
        plt.plot(steps, m_low, 'o-', color=color, label=f'{label} - Low')
        plt.plot(steps, m_high, 's--', color=color, label=f'{label} - High', alpha=0.6)
    
    plt.title(f'Cu_Step: {title_suffix} Mean Comparison', fontsize=15)
    plt.xlabel('Thickness Step (1-10)')
    plt.ylabel('Mean Intensity')
    plt.xticks(steps)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig(save_path)
    print(f"Summary plot saved to {save_path}")

def plot_adaptive_hist_grid(configs, channel_name, title_suffix, save_path):
    """Plots density histograms for all datasets in a 2x5 grid."""
    fig, axes = plt.subplots(2, 5, figsize=(22, 10), constrained_layout=True)
    axes = axes.flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for i in range(10):
        ax = axes[i]
        
        all_data_for_step = []
        for j, cfg in enumerate(configs):
            # idx 0 is low, idx 1 is high
            data_step = cfg['data'][0 if channel_name.lower()=='low' else 1][i]
            all_data_for_step.append(data_step)
            
        # Calculate common bin range for this step
        v_min = min(np.min(d) for d in all_data_for_step)
        v_max = max(np.max(d) for d in all_data_for_step)
        bins = np.linspace(v_min, v_max, 50)
        
        for j, cfg in enumerate(configs):
            ax.hist(all_data_for_step[j], bins=bins, alpha=0.4, 
                    label=cfg['label'], color=colors[j % 10], density=True)
        
        ax.set_title(f'Step {i+1}')
        if i == 0: ax.legend(loc='upper right', prop={'size': 9})
        ax.set_xlabel('Intensity')
    
    fig.suptitle(f'Cu_Step ({title_suffix}): {channel_name} Energy Histogram Comparison', fontsize=18)
    plt.savefig(save_path)
    print(f"Histogram grid saved to {save_path}")

def run_comparison(configs_desc, title_suffix, prefix):
    """Orchestrates the loading and plotting for a given set of comparisons."""
    configs = []
    for cd in configs_desc:
        print(f"Loading {cd['path']}...")
        # Auto-detect flip if not provided: usually Step 1 is the most intense
        low_raw, _ = load_dual_step_pixels(cd['path'], flip=False)
        should_flip = cd.get('flip')
        if should_flip is None:
            if np.mean(low_raw[0]) < np.mean(low_raw[-1]):
                print(f"  --> Auto-detected opposite gradient for {cd['label']}. Flipping.")
                should_flip = True
            else:
                should_flip = False
        
        low, high = load_dual_step_pixels(cd['path'], flip=should_flip)
        configs.append({'data': (low, high), 'label': cd['label']})
    
    os.makedirs('results/Tube_comparison', exist_ok=True)
    
    # 1. Unified Mean Plot
    plot_adaptive_means(configs, title_suffix, f'results/Tube_comparison/{prefix}_means.png')
    
    # 2. Histograms
    plot_adaptive_hist_grid(configs, "Low", title_suffix, f'results/Tube_comparison/{prefix}_hist_low.png')
    plot_adaptive_hist_grid(configs, "High", title_suffix, f'results/Tube_comparison/{prefix}_hist_high.png')

def main():
    # Setup plotting aesthetics for Chinese text
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False

    # TASK 1: Compare the three distinct equipment setups
    print("\n=== RUNNING 3-WAY EQUIPMENT COMPARISON ===")
    configs_3way = [
        {"path": r'results/20260331/pixel_values/160kV_4mA_step_sample_0_data.pkl', "label": "银山设备 (0331)"},
        {"path": r'results/20260407_Sample_test/pixel_values/Sample_160kV_test1_step_sample_1_data.pkl', "label": "院里设备 (0407)"},
        {"path": r'results/TYM_test/pixel_values/160kv-2mA-125us-0.5pF-disc-post_calib_step_sample_9_data.pkl', "label": "同源微 (0409)"}
    ]
    run_comparison(configs_3way, "3-Way Equipment", "CuStep_3Way")

    # TASK 2: Compare different exposure times (125us vs 270us) for TYM
    print("\n=== RUNNING EXPOSURE TIME COMPARISON (125us vs 270us) ===")
    configs_exposure = [
        {"path": r'results/TYM_test/pixel_values/160kv-2mA-125us-0.5pF-disc-post_calib_step_sample_9_data.pkl', "label": "125us (TYM)"},
        {"path": r'results/TYM_test/pixel_values/160kv-2mA-270us-0.5pF-disc-post_calib_step_sample_9_data.pkl', "label": "270us (TYM)"}
    ]
    run_comparison(configs_exposure, "Exposure Time", "TYM_Exposure")

if __name__ == "__main__":
    main()
