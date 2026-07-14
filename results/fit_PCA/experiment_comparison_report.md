# 实验结果综合对比报告

## 1. 工业选别指标对比 (全模型)

| 实验名称 | 模型 | 总体质量准确率 (Mass_Acc) | 产率 (Mass_Yield_Conc%) | 铜回收率 (Cu_Rec%) | 铜富集比 (Cu_ER) | 抛废率 (1-Yield) |
|---|---|---|---|---|---|---|
| exp_10_10_10 | PC2 Threshold (Count Acc) | 72.46% ± 7.04% | 0.23% | 0.02% | 0.10x | 99.77% |
| exp_10_10_10 | PC2 Threshold (Mass Acc) | 66.00% ± 10.23% | 8.69% | 1.74% | 0.20x | 91.31% |
| exp_10_10_10 | PC2 Threshold (Balanced Acc) | 42.15% ± 11.98% | 78.33% | 85.76% | 1.09x | 21.67% |
| exp_10_10_10 | Random Forest (RF) - PCA Fusion | 65.63% ± 8.47% | 11.25% | 11.53% | 1.03x | 88.75% |
| exp_10_10_10 | Gradient Boosting (GB) - PCA Fusion | 63.35% ± 9.77% | 20.70% | 20.74% | 1.00x | 79.30% |
| exp_10_10_10 | SVC - PCA Fusion | N/A | 35.97% | 40.59% | 1.13x | 64.03% |
| exp_metals_Z_le_30 | PC2 Threshold (Count Acc) | 72.46% ± 7.04% | 0.23% | 0.02% | 0.10x | 99.77% |
| exp_metals_Z_le_30 | PC2 Threshold (Mass Acc) | 69.19% ± 4.88% | 4.60% | 1.82% | 0.40x | 95.40% |
| exp_metals_Z_le_30 | PC2 Threshold (Balanced Acc) | 46.99% ± 16.23% | 70.38% | 76.72% | 1.09x | 29.62% |
| exp_metals_Z_le_30 | Random Forest (RF) - PCA Fusion | 64.81% ± 6.75% | 10.38% | 5.16% | 0.50x | 89.62% |
| exp_metals_Z_le_30 | Gradient Boosting (GB) - PCA Fusion | 59.82% ± 9.84% | 22.95% | 12.10% | 0.53x | 77.05% |
| exp_metals_Z_le_30 | SVC - PCA Fusion | N/A | 34.39% | 41.06% | 1.19x | 65.61% |
| exp_metals_Z_le_30_new_only | PC2 Threshold (Count Acc) | 72.56% ± 7.22% | 0.07% | 0.01% | 0.12x | 99.93% |
| exp_metals_Z_le_30_new_only | PC2 Threshold (Mass Acc) | 72.56% ± 7.22% | 0.07% | 0.01% | 0.12x | 99.93% |
| exp_metals_Z_le_30_new_only | PC2 Threshold (Balanced Acc) | 41.99% ± 16.27% | 54.17% | 57.19% | 1.06x | 45.83% |
| exp_metals_Z_le_30_new_only | Random Forest (RF) - PCA Fusion | 63.31% ± 9.55% | 11.48% | 4.65% | 0.41x | 88.52% |
| exp_metals_Z_le_30_new_only | Gradient Boosting (GB) - PCA Fusion | 57.45% ± 7.97% | 24.78% | 12.83% | 0.52x | 75.22% |
| exp_metals_Z_le_30_new_only | SVC - PCA Fusion | N/A | 41.82% | 47.78% | 1.14x | 58.18% |
| exp_metals_Z_le_30_new_only_greater_than | PC2 Threshold (Count Acc) | 70.86% ± 12.28% | 14.96% | 7.28% | 0.49x | 85.04% |
| exp_metals_Z_le_30_new_only_greater_than | PC2 Threshold (Mass Acc) | 65.86% ± 6.37% | 8.07% | 0.77% | 0.10x | 91.93% |
| exp_metals_Z_le_30_new_only_greater_than | PC2 Threshold (Balanced Acc) | 40.92% ± 11.32% | 78.58% | 84.52% | 1.08x | 21.42% |
| exp_metals_Z_le_30_new_only_greater_than | Random Forest (RF) - PCA Fusion | 63.31% ± 9.55% | 11.48% | 4.65% | 0.41x | 88.52% |
| exp_metals_Z_le_30_new_only_greater_than | Gradient Boosting (GB) - PCA Fusion | 57.45% ± 7.97% | 24.78% | 12.83% | 0.52x | 75.22% |
| exp_metals_Z_le_30_new_only_greater_than | SVC - PCA Fusion | N/A | 41.82% | 47.78% | 1.14x | 58.18% |

## 2. 子组质量准确率综合对比

下表反映了各模型在不同厚度、铁品位及铜品位区间下的详细分类表现。

### 实验: exp_10_10_10

**总体质量加权准确率 (Mass_Acc)**:
- **PC2 Threshold (Count Acc)**: 72.46% ± 7.04%
- **PC2 Threshold (Mass Acc)**: 66.00% ± 10.23%
- **PC2 Threshold (Balanced Acc)**: 42.15% ± 11.98%
- **Random Forest (RF)**: 60.22% ± 4.67%
- **Random Forest (RF) - PCA Fusion**: 65.63% ± 8.47%
- **Gradient Boosting (GB)**: 61.48% ± 8.02%
- **Gradient Boosting (GB) - PCA Fusion**: 63.35% ± 9.77%
- **Support Vector Machine (SVC)**: 42.06% ± 11.04%
- **Support Vector Machine (SVC) - PCA Fusion**: 52.03% ± 6.66%

| Thickness | Fe_Grade | Cu_Grade | Count | Total_Weight(g) | PC2 Threshold (Balanced Acc) | Random Forest (RF) - PCA Fusion | Gradient Boosting (GB) - PCA Fusion | SVC - PCA Fusion |
|---|---|---|---|---|---|---|---|---|
| Thin | Low Fe | Cu<0.05% | 58 | 1985.1 | 62.32% | 95.15% | 90.55% | 78.73% |
| Thin | Low Fe | Cu>0.1% | 20 | 698.8 | 40.03% | 12.15% | 17.22% | 16.33% |
| Thin | High Fe | Cu<0.05% | 44 | 1221.7 | 63.38% | 92.07% | 92.81% | 57.29% |
| Thin | High Fe | Cu>0.1% | 10 | 331.6 | 65.14% | 0.00% | 4.46% | 57.15% |
| Thick | Low Fe | Cu<0.05% | 41 | 3501.6 | 19.30% | 94.15% | 85.86% | 68.97% |
| Thick | Low Fe | Cu>0.1% | 15 | 1349.3 | 87.39% | 9.72% | 61.91% | 26.06% |
| Thick | High Fe | Cu<0.05% | 50 | 6574.1 | 4.75% | 83.14% | 77.84% | 52.95% |
| Thick | High Fe | Cu>0.1% | 31 | 3219.9 | 96.65% | 6.03% | 8.23% | 28.23% |

### 实验: exp_metals_Z_le_30

**总体质量加权准确率 (Mass_Acc)**:
- **PC2 Threshold (Count Acc)**: 72.46% ± 7.04%
- **PC2 Threshold (Mass Acc)**: 69.19% ± 4.88%
- **PC2 Threshold (Balanced Acc)**: 46.99% ± 16.23%
- **Random Forest (RF)**: 60.22% ± 4.67%
- **Random Forest (RF) - PCA Fusion**: 64.81% ± 6.75%
- **Gradient Boosting (GB)**: 61.48% ± 8.02%
- **Gradient Boosting (GB) - PCA Fusion**: 59.82% ± 9.84%
- **Support Vector Machine (SVC)**: 42.06% ± 11.04%
- **Support Vector Machine (SVC) - PCA Fusion**: 53.83% ± 8.59%

| Thickness | Fe_Grade | Cu_Grade | Count | Total_Weight(g) | PC2 Threshold (Balanced Acc) | Random Forest (RF) - PCA Fusion | Gradient Boosting (GB) - PCA Fusion | SVC - PCA Fusion |
|---|---|---|---|---|---|---|---|---|
| Thin | Low Fe | Cu<0.05% | 58 | 1985.1 | 81.02% | 92.69% | 91.50% | 72.93% |
| Thin | Low Fe | Cu>0.1% | 20 | 698.8 | 23.70% | 12.15% | 12.15% | 23.28% |
| Thin | High Fe | Cu<0.05% | 44 | 1221.7 | 77.93% | 98.28% | 85.61% | 57.29% |
| Thin | High Fe | Cu>0.1% | 10 | 331.6 | 65.14% | 0.00% | 6.03% | 62.82% |
| Thick | Low Fe | Cu<0.05% | 41 | 3501.6 | 30.50% | 91.45% | 81.61% | 68.97% |
| Thick | Low Fe | Cu>0.1% | 15 | 1349.3 | 73.16% | 9.72% | 54.08% | 26.06% |
| Thick | High Fe | Cu<0.05% | 50 | 6574.1 | 7.80% | 84.07% | 60.48% | 60.68% |
| Thick | High Fe | Cu>0.1% | 31 | 3219.9 | 94.03% | 0.00% | 2.25% | 28.23% |

### 实验: exp_metals_Z_le_30_new_only

**总体质量加权准确率 (Mass_Acc)**:
- **PC2 Threshold (Count Acc)**: 72.56% ± 7.22%
- **PC2 Threshold (Mass Acc)**: 72.56% ± 7.22%
- **PC2 Threshold (Balanced Acc)**: 41.99% ± 16.27%
- **Random Forest (RF)**: 60.22% ± 4.67%
- **Random Forest (RF) - PCA Fusion**: 63.31% ± 9.55%
- **Gradient Boosting (GB)**: 61.48% ± 8.02%
- **Gradient Boosting (GB) - PCA Fusion**: 57.45% ± 7.97%
- **Support Vector Machine (SVC)**: 42.06% ± 11.04%
- **Support Vector Machine (SVC) - PCA Fusion**: 52.37% ± 10.85%

| Thickness | Fe_Grade | Cu_Grade | Count | Total_Weight(g) | PC2 Threshold (Balanced Acc) | Random Forest (RF) - PCA Fusion | Gradient Boosting (GB) - PCA Fusion | SVC - PCA Fusion |
|---|---|---|---|---|---|---|---|---|
| Thin | Low Fe | Cu<0.05% | 58 | 1985.1 | 0.91% | 94.11% | 89.71% | 80.53% |
| Thin | Low Fe | Cu>0.1% | 20 | 698.8 | 100.00% | 0.00% | 18.63% | 53.13% |
| Thin | High Fe | Cu<0.05% | 44 | 1221.7 | 5.93% | 98.28% | 88.14% | 51.17% |
| Thin | High Fe | Cu>0.1% | 10 | 331.6 | 100.00% | 18.64% | 0.00% | 58.05% |
| Thick | Low Fe | Cu<0.05% | 41 | 3501.6 | 35.49% | 81.82% | 73.02% | 53.45% |
| Thick | Low Fe | Cu>0.1% | 15 | 1349.3 | 41.74% | 3.60% | 50.26% | 39.33% |
| Thick | High Fe | Cu<0.05% | 50 | 6574.1 | 66.34% | 77.42% | 59.92% | 54.06% |
| Thick | High Fe | Cu>0.1% | 31 | 3219.9 | 30.00% | 1.06% | 4.47% | 33.74% |

### 实验: exp_metals_Z_le_30_new_only_greater_than

**总体质量加权准确率 (Mass_Acc)**:
- **PC2 Threshold (Count Acc)**: 70.86% ± 12.28%
- **PC2 Threshold (Mass Acc)**: 65.86% ± 6.37%
- **PC2 Threshold (Balanced Acc)**: 40.92% ± 11.32%
- **Random Forest (RF)**: 60.22% ± 4.67%
- **Random Forest (RF) - PCA Fusion**: 63.31% ± 9.55%
- **Gradient Boosting (GB)**: 61.48% ± 8.02%
- **Gradient Boosting (GB) - PCA Fusion**: 57.45% ± 7.97%
- **Support Vector Machine (SVC)**: 42.06% ± 11.04%
- **Support Vector Machine (SVC) - PCA Fusion**: 52.37% ± 10.85%

| Thickness | Fe_Grade | Cu_Grade | Count | Total_Weight(g) | PC2 Threshold (Balanced Acc) | Random Forest (RF) - PCA Fusion | Gradient Boosting (GB) - PCA Fusion | SVC - PCA Fusion |
|---|---|---|---|---|---|---|---|---|
| Thin | Low Fe | Cu<0.05% | 58 | 1985.1 | 60.47% | 94.11% | 89.71% | 80.53% |
| Thin | Low Fe | Cu>0.1% | 20 | 698.8 | 28.73% | 0.00% | 18.63% | 53.13% |
| Thin | High Fe | Cu<0.05% | 44 | 1221.7 | 63.78% | 98.28% | 88.14% | 51.17% |
| Thin | High Fe | Cu>0.1% | 10 | 331.6 | 50.09% | 18.64% | 0.00% | 58.05% |
| Thick | Low Fe | Cu<0.05% | 41 | 3501.6 | 13.66% | 81.82% | 73.02% | 53.45% |
| Thick | Low Fe | Cu>0.1% | 15 | 1349.3 | 92.77% | 3.60% | 50.26% | 39.33% |
| Thick | High Fe | Cu<0.05% | 50 | 6574.1 | 5.15% | 77.42% | 59.92% | 54.06% |
| Thick | High Fe | Cu>0.1% | 31 | 3219.9 | 94.99% | 1.06% | 4.47% | 33.74% |

