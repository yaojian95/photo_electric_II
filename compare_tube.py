import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

def load_dual_step_pixels(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    if not isinstance(data['pixels_low'], list):
        raise ValueError(f"File {file_path} does not contain a list of steps.")
    return data['pixels_low'], data['pixels_high']

def plot_all_means(low1, high1, low2, high2, low3, high3, label1, label2, label3, filename):
    plt.figure(figsize=(12, 7))
    steps = np.arange(1, 11)
    
    m_low1, m_high1 = [np.mean(p) for p in low1], [np.mean(p) for p in high1]
    m_low2, m_high2 = [np.mean(p) for p in low2], [np.mean(p) for p in high2]
    m_low3, m_high3 = [np.mean(p) for p in low3], [np.mean(p) for p in high3]
    
    plt.plot(steps, m_low1, 'o-', color='blue', label=f'{label1} - Low')
    plt.plot(steps, m_high1, 's--', color='blue', label=f'{label1} - High', alpha=0.6)
    
    plt.plot(steps, m_low2, 'o-', color='orange', label=f'{label2} - Low')
    plt.plot(steps, m_high2, 's--', color='orange', label=f'{label2} - High', alpha=0.6)
    
    plt.plot(steps, m_low3, 'o-', color='green', label=f'{label3} - Low')
    plt.plot(steps, m_high3, 's--', color='green', label=f'{label3} - High', alpha=0.6)
    
    plt.title('Cu_Step: Unified 3-Way Mean Intensity Comparison', fontsize=15)
    plt.xlabel('Thickness Step (1-10)')
    plt.ylabel('Mean Intensity')
    plt.xticks(steps)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig(filename)
    print(f"Unified plot saved to {filename}")

def plot_intensity_grid(px1, px2, px3, title_prefix, filename, label1, label2, label3):
    fig, axes = plt.subplots(2, 5, figsize=(22, 10), constrained_layout=True)
    axes = axes.flatten()
    
    for i in range(10):
        ax = axes[i]
        d1, d2, d3 = px1[i], px2[i], px3[i]
        
        # Calculate common bin range
        v_min = min(np.min(d1), np.min(d2), np.min(d3))
        v_max = max(np.max(d1), np.max(d2), np.max(d3))
        bins = np.linspace(v_min, v_max, 50)
        
        ax.hist(d1, bins=bins, alpha=0.4, label=label1, color='blue', density=True)
        ax.hist(d2, bins=bins, alpha=0.4, label=label2, color='orange', density=True)
        ax.hist(d3, bins=bins, alpha=0.4, label=label3, color='green', density=True)
        
        ax.set_title(f'Step {i+1}')
        if i == 0: ax.legend(loc='upper right', prop={'size': 9})
        ax.set_xlabel('Intensity')
    
    fig.suptitle(f'Cu_Step: {title_prefix} 3-Way Histogram Comparison', fontsize=18)
    plt.savefig(filename)
    print(f"Histogram grid saved to {filename}")

def main():
    # Define paths
    path1 = r'results/20260331/pixel_values/160kV_4mA_step_sample_0_data.pkl'
    path2 = r'results/20260407_Sample_test/pixel_values/Sample_160kV_test1_step_sample_1_data.pkl'
    path3 = r'results/TYM_test/pixel_values/160kv-2mA-125us-0.5pF-disc-post_calib_step_sample_9_data.pkl'
    
    label1, label2, label3 = "银山设备 (0331)", "院里设备 (0407)", "同源微 (0409)"
    
    # Ensure output directory exists
    os.makedirs('results/Tube_comparison', exist_ok=True)
    
    print(f"Loading {path1}...")
    low1, high1 = load_dual_step_pixels(path1)
    
    print(f"Loading {path2}...")
    low2, high2 = load_dual_step_pixels(path2)
    
    print(f"Loading {path3}...")
    low3, high3 = load_dual_step_pixels(path3)
    
    # Align thickness:
    # Dataset 1 (0331) is reference (Step 1 is brightest). 
    # Dataset 2 (0407) was found to be opposite.
    low2_aligned = low2[::-1]
    high2_aligned = high2[::-1]
    
    # For Dataset 3 (0409), check first few values or just provide alignment if requested.
    # Usually these samples follow a consistent pattern. If Step 1 is darkest, flip it.
    if np.mean(low3[0]) < np.mean(low3[-1]):
        print("Detected opposite gradient in Dataset 3 (0409). Aligned.")
        low3_aligned = low3[::-1]
        high3_aligned = high3[::-1]
    else:
        low3_aligned = low3
        high3_aligned = high3
    
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False
    
    plot_all_means(low1, high1, low2_aligned, high2_aligned, low3_aligned, high3_aligned, 
                   label1, label2, label3, 'results/Tube_comparison/CuStep_Summary_Means.png')
    
    plot_intensity_grid(low1, low2_aligned, low3_aligned, 
                        "Low Energy", 'results/Tube_comparison/CuStep_Comparison_Hist_Low.png', 
                        label1, label2, label3)
    
    plot_intensity_grid(high1, high2_aligned, high3_aligned, 
                        "High Energy", 'results/Tube_comparison/CuStep_Comparison_Hist_High.png', 
                        label1, label2, label3)

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
