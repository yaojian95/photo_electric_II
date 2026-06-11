import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
STEPS_DIR = ROOT_DIR / "steps"
BASIS_DIR = ROOT_DIR / "data" / "normalization"
MU_DIR = ROOT_DIR / "data" / "mass_atten"
OUT_DIR = ROOT_DIR / "invsp_results"

I0_VALUE = 52428.0
FILTERS = ("0.6", "1.2")
VOLTAGES = (200, 220, 240, 260, 280, 300, 320)

MATERIALS = {
    "铜": {
        "key": "Cu",
        "density": 8.96,
        "thicknesses_mm": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    },
    "铁": {
        "key": "Fe",
        "density": 7.874,
        "thicknesses_mm": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    },
    "铝": {
        "key": "Al",
        "density": 2.70,
        "thicknesses_mm": [12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
    },
}


@dataclass
class StepSample:
    material_cn: str
    material_key: str
    kvp: int
    filter_mm: str
    band: str
    thickness_mm: float
    mean_value: float
    ratio: float
    image_path: Path
    mask_path: Path


def _read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise FileNotFoundError(f"无法读取文件: {path}")
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"OpenCV 无法解码文件: {path}")
    return img


def load_basis_spectra(csv_path: Path) -> Tuple[np.ndarray, List[str], np.ndarray]:
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = list(reader)

    energy = np.array([float(row[0]) for row in rows], dtype=np.float64)
    labels = headers[1:]
    basis = np.array([[float(v) for v in row[1:]] for row in rows], dtype=np.float64)

    col_sum = np.sum(basis, axis=0, keepdims=True)
    col_sum = np.where(col_sum <= 0.0, 1.0, col_sum)
    basis = basis / col_sum
    return energy, labels, basis


def load_all_basis_spectra(common_energy: np.ndarray, max_kvp: int, basis_mode: str = "all") -> Tuple[np.ndarray, List[str], np.ndarray]:
    """加载 normalization 下能谱文件，插值到统一能量网格后合并。

    basis_mode="all"  : 加载所有电压 ≤ max_kvp 的基谱
    basis_mode="match": 仅加载电压 == max_kvp 的基谱

    返回 (energy, labels, basis)，basis 形状为 (n_energy, n_total_spectra)。
    """
    all_labels: List[str] = []
    all_spectra: List[np.ndarray] = []
    for csv_path in sorted(BASIS_DIR.glob("spectrum_*kV.csv")):
        kv_str = csv_path.stem.replace("spectrum_", "").replace("kV", "")
        try:
            file_kv = int(kv_str)
        except ValueError:
            continue
        if basis_mode == "all" and file_kv > max_kvp:
            continue
        elif basis_mode == "match" and file_kv != max_kvp:
            continue
        energy, labels, basis = load_basis_spectra(csv_path)
        # 插值到统一能量网格
        spec_interp = np.zeros((len(common_energy), len(labels)), dtype=np.float64)
        for j in range(len(labels)):
            spec_interp[:, j] = np.interp(common_energy, energy, basis[:, j], left=0.0, right=0.0)
        # 重新归一化
        col_sum = np.sum(spec_interp, axis=0, keepdims=True)
        col_sum = np.where(col_sum <= 0.0, 1.0, col_sum)
        spec_interp /= col_sum
        all_labels.extend([f"{csv_path.stem.replace('spectrum_', '')}_{lbl}" for lbl in labels])
        all_spectra.append(spec_interp)
    merged = np.hstack(all_spectra)
    return common_energy, all_labels, merged


def load_mass_atten(csv_path: Path) -> Dict[str, np.ndarray]:
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = list(reader)

    result: Dict[str, np.ndarray] = {"energy_keV": np.array([float(r[0]) for r in rows], dtype=np.float64)}
    for j, name in enumerate(headers[1:], start=1):
        result[name] = np.array([float(r[j]) for r in rows], dtype=np.float64)
    return result


def find_step_pair(material_cn: str, kvp: int, filter_mm: str) -> Tuple[Path, Path]:
    patterns = [
        f"{material_cn}阶梯-*-{filter_mm}mm-{kvp}kV-2mA-orig.png",
        f"{material_cn}阶梯-*-{filter_mm}mm-{kvp}kV-2mA-user.png",
    ]

    for pattern in patterns:
        matches = sorted(STEPS_DIR.glob(pattern))
        if not matches:
            continue
        mask_path = matches[0]
        image_path = mask_path.with_suffix(".tif")
        if image_path.exists():
            return image_path, mask_path

    raise FileNotFoundError(
        f"未找到阶梯图像: material={material_cn}, kv={kvp}, filter={filter_mm}"
    )


def extract_step_samples(
    image_path: Path,
    mask_path: Path,
    material_cn: str,
    material_key: str,
    kvp: int,
    filter_mm: str,
    thicknesses_mm: List[float],
) -> List[StepSample]:
    image = _read_image(image_path)
    mask = _read_image(mask_path)

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    binary = (mask > 0).astype(np.uint8)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components = []
    for idx in range(1, n_labels):
        x, y, w, h, area = stats[idx]
        cx, cy = centroids[idx]
        components.append(
            {
                "label": idx,
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "area": int(area),
                "cx": float(cx),
                "cy": float(cy),
            }
        )

    if len(components) != 20:
        raise ValueError(f"{mask_path.name} 连通区域不是 20 个，而是 {len(components)} 个")

    mid_x = float(np.median([c["cx"] for c in components]))
    low_components = [c for c in components if c["cx"] < mid_x]
    high_components = [c for c in components if c["cx"] >= mid_x]
    if len(low_components) != 10 or len(high_components) != 10:
        raise ValueError(
            f"{mask_path.name} 左右区域数量异常: low={len(low_components)}, high={len(high_components)}"
        )

    low_components.sort(key=lambda item: item["cy"], reverse=True)
    high_components.sort(key=lambda item: item["cy"], reverse=True)

    samples: List[StepSample] = []
    for band, band_components in (("low", low_components), ("high", high_components)):
        for thickness_mm, comp in zip(thicknesses_mm, band_components):
            region_mask = labels == comp["label"]
            mean_value = float(np.mean(image[region_mask]))
            ratio = mean_value / I0_VALUE
            samples.append(
                StepSample(
                    material_cn=material_cn,
                    material_key=material_key,
                    kvp=kvp,
                    filter_mm=filter_mm,
                    band=band,
                    thickness_mm=float(thickness_mm),
                    mean_value=mean_value,
                    ratio=ratio,
                    image_path=image_path,
                    mask_path=mask_path,
                )
            )
    return samples


def build_material_matrix(
    basis: np.ndarray,
    mu_mass: np.ndarray,
    density: float,
    thicknesses_mm: List[float],
) -> np.ndarray:
    d_cm = np.asarray(thicknesses_mm, dtype=np.float64) / 10.0
    mu_linear = mu_mass * density
    transmission = np.exp(-np.outer(mu_linear, d_cm))
    return transmission.T @ basis


def _project_to_simplex(x: np.ndarray, eps: float) -> np.ndarray:
    x = np.maximum(np.asarray(x, dtype=np.float64), eps)
    s = float(np.sum(x))
    if s <= 0.0:
        return np.full_like(x, 1.0 / x.size)
    return x / s


def solve_cmd(
    A: np.ndarray,
    b: np.ndarray,
    n_iter: int = 4000,
    tol: float = 1e-15,
    eps: float = 1e-15,
    dt: float = 0.0,
    c0: np.ndarray = None,
) -> Tuple[np.ndarray, Dict[str, List[float]]]:
    """An & Hou, Phys. Rev. A 77, 042702 (2008) — CMD 优化方法求解 c_n。

    Eq. (11):  M_i  d²c_i/dτ²  =  −∂D/∂c_i
    Eq. (12):  M_i  =  Σ_m Σ_{n≠i} A_mn   (虚拟原子质量)

    目标函数: D(c) = ½ Σ_m (Σ_n A_mn c_n − b_m)²   (系统势能)

    通过速度 Verlet 积分 + 零温阻尼 (v·f ≥ 0 ⇒ v=0) 将系统弛豫到
    势能最小点, 同时强制 c_n ≥ 0, Σ c_n = 1。

    dt=0 时自动根据问题规模计算合适的虚拟时间步长。
    步长按余弦退火衰减: dt_k = dt * 0.5 * (1 + cos(pi * k / n_iter))。
    """
    N = A.shape[1]

    # --- Eq. (12): 虚拟原子质量 μ_s = total_sum − col_sum_s ---
    col_sum = np.sum(A, axis=0)               # Σ_m A_ms
    total_sum = float(np.sum(col_sum))        # Σ_{m,s} A_ms
    mass = np.maximum(total_sum - col_sum, eps)

    def grad(x: np.ndarray) -> np.ndarray:
        """∂D/∂c = A^T (A c − b)"""
        return A.T @ (A @ x - b)

    # --- 初始化: c = c0 或均匀分布, v = -f·dt/m ---
    if c0 is not None:
        c = _project_to_simplex(np.asarray(c0, dtype=np.float64), eps)
    else:
        c = np.full(N, 1.0 / N, dtype=np.float64)
    f = -grad(c)                               # f_s = −∂D/∂c_s

    # --- 自动缩放 dt：使第一步位移 ~ 1/N ---
    if dt <= 0.0:
        a_max = float(np.max(np.abs(f / mass)))
        if a_max > 0:
            dt = float(np.sqrt(1.0 / N / a_max))
        else:
            dt = 1e-4

    dt0 = dt                                    # 保存初始步长用于余弦退火
    v = -f * dt0 / mass                        # 初始半步步速度

    history: Dict[str, List[float]] = {"rmse": [], "objective": [], "kinetic": []}

    for k in range(n_iter):
        # --- 余弦退火步长 ---
        dt_k = dt0 * 0.5 * (1.0 + np.cos(np.pi * k / n_iter))

        # --- Verlet 位置更新 ---
        c_new = c + dt_k * v + 0.5 * (dt_k ** 2) * (f / mass)
        c_new = _project_to_simplex(c_new, eps)

        # --- 新力 ---
        f_new = -grad(c_new)

        # --- Verlet 速度更新 ---
        v_new = v + 0.5 * dt_k * ((f + f_new) / mass)

        # --- 零温阻尼: v_s · f_s ≥ 0 时置零 ---
        same_sign = v_new * f_new >= 0.0
        v_new[same_sign] = 0.0

        # --- 诊断 ---
        pred = A @ c_new
        rmse = float(np.sqrt(np.mean((pred - b) ** 2)))
        obj = float(0.5 * np.mean((pred - b) ** 2))
        ke = float(0.5 * np.sum(mass * v_new ** 2))
        history["rmse"].append(rmse)
        history["objective"].append(obj)
        history["kinetic"].append(ke)

        # --- 收敛判据 ---
        if k >= 20 and np.linalg.norm(c_new - c, ord=1) < tol:
            c = c_new
            break

        c = c_new
        v = v_new
        f = f_new

    return c, history

def _solve_cg(
    A: np.ndarray, b: np.ndarray, n_iter: int,
    eps: float,
    c0: np.ndarray = None,
) -> Tuple[np.ndarray, Dict[str, List[float]]]:
    """投影非线性共轭梯度 (Fletcher-Reeves) 求解 min ½||Ac-b||², s.t. c≥0, Σc=1。"""
    M = A.shape[1]
    c = np.asarray(c0, dtype=np.float64) if c0 is not None else np.full(M, 1.0 / M, dtype=np.float64)
    c = _project_to_simplex(c, eps)
    history: Dict[str, List[float]] = {"rmse": [], "objective": [], "kinetic": []}

    def _grad(x: np.ndarray) -> np.ndarray:
        return A.T @ (A @ x - b)

    g = _grad(c)
    d = -g.copy()

    for k in range(n_iter):
        # 线搜索: Armijo 回溯
        alpha = 1.0
        c_new = _project_to_simplex(c + alpha * d, eps)
        f_old = 0.5 * np.mean((A @ c - b) ** 2)
        f_new = 0.5 * np.mean((A @ c_new - b) ** 2)
        for _ in range(20):
            if f_new <= f_old + 1e-4 * alpha * float(np.dot(g, d)):
                break
            alpha *= 0.5
            c_new = _project_to_simplex(c + alpha * d, eps)
            f_new = 0.5 * np.mean((A @ c_new - b) ** 2)

        g_new = _grad(c_new)
        # Fletcher-Reeves beta, 带 Powell 重启
        if k % M == 0 or float(np.dot(g_new, g)) < 0:
            beta = 0.0
        else:
            beta = float(np.dot(g_new, g_new) / max(np.dot(g, g), 1e-30))

        d = -g_new + beta * d
        c = c_new
        g = g_new

        pred = A @ c
        rmse = float(np.sqrt(np.mean((pred - b) ** 2)))
        obj = float(0.5 * np.mean((pred - b) ** 2))
        history["rmse"].append(rmse)
        history["objective"].append(obj)
        history["kinetic"].append(0.0)

    return c, history


def solve_weights(
    A: np.ndarray, b: np.ndarray,
    method: str = "cmd",
    n_iter: int = 10000,
    dt: float = 0.0,
    eps: float = 1e-15,
    warmup: int = 0,
) -> Tuple[np.ndarray, Dict[str, List[float]]]:
    """统一求解入口。

    method: "cmd" | "cg"
    warmup: 先用 CMD 预热迭代次数 (0=跳过, 所有方法通用)
    """
    # ── CMD 预热（所有方法通用）──
    c0 = None
    if warmup > 0:
        c0, _ = solve_cmd(A=A, b=b, n_iter=warmup, dt=dt, eps=eps)

    if method == "cmd":
        return solve_cmd(A=A, b=b, n_iter=n_iter, dt=dt, eps=eps, c0=c0)
    elif method == "cg":
        return _solve_cg(A=A, b=b, n_iter=n_iter, eps=eps, c0=c0)
    else:
        raise ValueError(f"未知求解方法: {method}，可选 cmd/cg")


def prepare_measurements(kvp: int) -> List[StepSample]:
    all_samples: List[StepSample] = []
    for material_cn, config in MATERIALS.items():
        for filter_mm in FILTERS:
            image_path, mask_path = find_step_pair(material_cn, kvp, filter_mm)
            samples = extract_step_samples(
                image_path=image_path,
                mask_path=mask_path,
                material_cn=material_cn,
                material_key=config["key"],
                kvp=kvp,
                filter_mm=filter_mm,
                thicknesses_mm=config["thicknesses_mm"],
            )
            all_samples.extend(samples)
    return all_samples


def build_problem(
    basis: np.ndarray,
    mu_map: Dict[str, np.ndarray],
    samples: List[StepSample],
    fit_n: int = 0,
    fit_mode: str = "thick",
    weight_power: float = 0.0,
    material_weights: Dict[str, float] = None,
) -> Tuple[np.ndarray, np.ndarray, List[StepSample]]:
    """构建 A 矩阵和 b 向量。

    fit_n > 0 时每种材料选取 fit_n 个厚度（而非样本数）进行拟合:
      fit_mode="thin"  : 取最薄的前 fit_n 个厚度
      fit_mode="thick" : 取最厚的后 fit_n 个厚度
    同一厚度下的所有样本（如不同滤片）均参与训练。
    fit_n=0 使用全部厚度（用于预测）。

    weight_power > 0 时给薄厚度更高权重: w_i ∝ (1/thickness)^power,
    material_weights 给不同材料额外的权重因子 (e.g. {"Cu": 0.2, "Fe": 0.1, "Al": 0.7}),
    最终 w_i = material_weight × thickness^(-power),
    通过 sqrt(w) 缩放 A 和 b 行实现加权 MSE: ½ Σ w_i (a_i·c − b_i)²。
    """
    rows: List[np.ndarray] = []
    values: List[float] = []
    ordered_samples: List[StepSample] = []

    grouped: Dict[str, List[StepSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.material_key, []).append(sample)

    for material_cn, config in MATERIALS.items():
        material_key = config["key"]
        material_samples = grouped.get(material_key, [])
        material_samples.sort(key=lambda s: s.thickness_mm)
        if not material_samples:
            continue

        # 选取 fit_n 个厚度（而非样本数）用于训练
        if fit_n > 0:
            unique_thicknesses = sorted(set(s.thickness_mm for s in material_samples))
            if fit_mode == "thin":
                selected_thicknesses = set(unique_thicknesses[:fit_n])
            else:
                selected_thicknesses = set(unique_thicknesses[-fit_n:])
            material_samples = [s for s in material_samples if s.thickness_mm in selected_thicknesses]

        A_material = build_material_matrix(
            basis=basis,
            mu_mass=mu_map[f"{material_key}_mu_cm2_g"],
            density=config["density"],
            thicknesses_mm=[s.thickness_mm for s in material_samples],
        )
        for row, sample in zip(A_material, material_samples):
            rows.append(row)
            values.append(sample.ratio)
            ordered_samples.append(sample)

    A = np.vstack(rows)
    b = np.array(values, dtype=np.float64)

    # 样本加权: 薄厚度权重更大 + 材料权重
    if weight_power > 0.0 or material_weights:
        thicknesses = np.array([s.thickness_mm for s in ordered_samples], dtype=np.float64)
        w = np.ones(len(ordered_samples), dtype=np.float64)
        if weight_power > 0.0:
            w = w * (thicknesses ** (-weight_power))
        if material_weights:
            for i, s in enumerate(ordered_samples):
                w[i] = w[i] * material_weights.get(s.material_key, 1.0)
        w = w / np.mean(w)                    # 归一化到均值=1
        sqrt_w = np.sqrt(w)
        A = A * sqrt_w[:, np.newaxis]         # 每行乘以 sqrt(w_i)
        b = b * sqrt_w

    return A, b, ordered_samples


def save_weights(path: Path, labels: List[str], weight_map: Dict[str, np.ndarray]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["basis_label"] + list(weight_map.keys()))
        for idx, label in enumerate(labels):
            writer.writerow([label] + [f"{weight_map[key][idx]:.10e}" for key in weight_map])


def save_spectra(path: Path, energy: np.ndarray, spectra: Dict[str, np.ndarray]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["energy_keV"] + list(spectra.keys()))
        for i, e in enumerate(energy):
            writer.writerow([f"{e:.10e}"] + [f"{spectra[key][i]:.10e}" for key in spectra])


def save_fit(path: Path, fit_rows: List[Dict[str, object]]) -> None:
    headers = [
        "problem",
        "material",
        "filter_mm",
        "band",
        "thickness_mm",
        "measured_ratio",
        "fitted_ratio",
        "abs_error_ratio",
        "image_name",
        "mask_name",
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in fit_rows:
            writer.writerow(row)


def run_voltage(
    kvp: int, n_iter: int, dt: float, fit_n: int,
    basis_mode: str = "all", fit_mode: str = "thick",
    method: str = "cmd",
    warmup: int = 0, weight_power: float = 0.0,
    material_weights: Dict[str, float] = None,
) -> None:
    print(f"\n{'=' * 72}")
    print(f"处理电压: {kvp} kV, fit_n={fit_n}, basis_mode={basis_mode}, fit_mode={fit_mode}, method={method}, weight_power={weight_power}, material_weights={material_weights}")

    # 加载最大电压的能谱获取统一能量网格 (1.5~319.5 keV, 40 bins)
    max_kv_energy, _, _ = load_basis_spectra(BASIS_DIR / "spectrum_320kV.csv")
    # 加载 mu 并插值到统一网格
    mu_raw = load_mass_atten(MU_DIR / "mu_320kV.csv")
    mu_map: Dict[str, np.ndarray] = {}
    for key in ["Cu_mu_cm2_g", "Fe_mu_cm2_g", "Al_mu_cm2_g"]:
        mu_map[key] = np.interp(max_kv_energy, mu_raw["energy_keV"], mu_raw[key])

    # 加载基谱（basis_mode 控制：all=所有≤kvp, match=仅等于kvp）
    energy, basis_labels, basis = load_all_basis_spectra(max_kv_energy, kvp, basis_mode)
    if basis_mode == "all":
        n_voltage_files = len([f for f in BASIS_DIR.glob("spectrum_*kV.csv") if int(f.stem.replace("spectrum_","").replace("kV","")) <= kvp])
    else:
        n_voltage_files = 1
    print(f"基谱数量: {len(basis_labels)} (来自 {n_voltage_files} 个电压 ≤ {kvp} kV), 能量点: {len(energy)}")

    samples = prepare_measurements(kvp)
    low_samples = [s for s in samples if s.band == "low"]
    high06_samples = [s for s in samples if s.band == "high" and s.filter_mm == "0.6"]
    high12_samples = [s for s in samples if s.band == "high" and s.filter_mm == "1.2"]
    print(
        f"测量点数量: low={len(low_samples)}, high_0.6={len(high06_samples)}, high_1.2={len(high12_samples)}"
    )

    problems = {
        "low": low_samples,
        "high_0.6mm": high06_samples,
        "high_1.2mm": high12_samples,
    }

    weight_map: Dict[str, np.ndarray] = {}
    spectra_map: Dict[str, np.ndarray] = {}
    fit_rows: List[Dict[str, object]] = []

    for problem_name, problem_samples in problems.items():
        # fit_n 个厚度用于训练（weight_power 给薄厚度更高权重, material_weights 按材料加权）
        A_train, b_train, _ = build_problem(basis=basis, mu_map=mu_map, samples=problem_samples, fit_n=fit_n, fit_mode=fit_mode, weight_power=weight_power, material_weights=material_weights)
        weights, history = solve_weights(
            A=A_train, b=b_train, method=method, n_iter=n_iter, dt=dt,
            warmup=warmup,
        )

        # 全部厚度用于预测（不加权，报告真实误差）
        A_full, b_full, ordered_samples = build_problem(basis=basis, mu_map=mu_map, samples=problem_samples)
        fitted_full = A_full @ weights
        spectrum = basis @ weights
        spectrum = spectrum / max(np.sum(spectrum), 1e-12)

        weight_map[problem_name] = weights
        spectra_map[problem_name] = spectrum

        rmse_train = float(np.sqrt(np.mean((A_train @ weights - b_train) ** 2)))
        rmse_full = float(np.sqrt(np.mean((fitted_full - b_full) ** 2)))
        nnz = int(np.sum(weights > 1e-12))
        print(f"  {problem_name}: train_rows={A_train.shape[0]}, train_rmse={rmse_train:.6e}, "
              f"full_rmse={rmse_full:.6e}, iters={len(history['rmse'])}, nonzero={nnz}")

        for sample, fitted_val in zip(ordered_samples, fitted_full):
            fit_rows.append(
                {
                    "problem": problem_name,
                    "material": sample.material_key,
                    "filter_mm": sample.filter_mm,
                    "band": sample.band,
                    "thickness_mm": f"{sample.thickness_mm:.1f}",
                    "measured_ratio": f"{sample.ratio:.10e}",
                    "fitted_ratio": f"{float(fitted_val):.10e}",
                    "abs_error_ratio": f"{abs(float(fitted_val) - sample.ratio):.10e}",
                    "image_name": sample.image_path.name,
                    "mask_name": sample.mask_path.name,
                }
            )

    out_dir = OUT_DIR / f"{kvp}kV"
    out_dir.mkdir(exist_ok=True, parents=True)

    save_weights(out_dir / "weights.csv", basis_labels, weight_map)
    save_spectra(out_dir / "reconstructed_spectra.csv", energy, spectra_map)
    save_fit(out_dir / "fit.csv", fit_rows)
    print(f"结果已保存到: {out_dir}")

    # ── 可视化 ──────────────────────────────────────────────────────
    _plot_results(
        kvp=kvp,
        energy=energy,
        basis=basis,
        mu_map=mu_map,
        spectra_map=spectra_map,
        weight_map=weight_map,
        problems=problems,
        out_dir=out_dir,
    )


def _plot_results(
    kvp: int,
    energy: np.ndarray,
    basis: np.ndarray,
    mu_map: Dict[str, np.ndarray],
    spectra_map: Dict[str, np.ndarray],
    weight_map: Dict[str, np.ndarray],
    problems: Dict[str, List[StepSample]],
    out_dir: Path,
) -> None:
    """可视化重建能谱及拟合 vs 实测对比。"""
    # ── Figure 1: 重建的高/低能输入能谱 ──
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    colors = {"low": "#1f77b4", "high_0.6mm": "#d62728", "high_1.2mm": "#ff7f0e"}
    labels = {"low": "low spectrum (0.6/1.2mm filter shared)",
              "high_0.6mm": "high spectrum (0.6mm filter)",
              "high_1.2mm": "high spectrum (1.2mm filter)"}
    for name, spectrum in spectra_map.items():
        ax1.plot(energy, spectrum, '-', color=colors[name], linewidth=1.8, label=labels.get(name, name))
    ax1.set_xlabel("Energy (keV)")
    ax1.set_ylabel("Normalized fluence")
    ax1.set_title(f"{kvp} kV — reconstructed incident spectra")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(energy[0], energy[-1])
    fig1.tight_layout()
    fig1.savefig(str(out_dir / f"reconstructed_spectra_{kvp}kV.png"), dpi=150)
    print(f"  Saved chart: {out_dir / f'reconstructed_spectra_{kvp}kV.png'}")

    # ── Figure 2: 测量数据 vs 拟合透射曲线 ──
    mat_configs = [
        ("Cu", MATERIALS["铜"]["density"], MATERIALS["铜"]["thicknesses_mm"]),
        ("Fe", MATERIALS["铁"]["density"], MATERIALS["铁"]["thicknesses_mm"]),
        ("Al", MATERIALS["铝"]["density"], MATERIALS["铝"]["thicknesses_mm"]),
    ]
    prob_names = ["low", "high_0.6mm", "high_1.2mm"]
    prob_colors = {"low": "#1f77b4", "high_0.6mm": "#d62728", "high_1.2mm": "#ff7f0e"}
    prob_labels = {"low": "low", "high_0.6mm": "high 0.6mm", "high_1.2mm": "high 1.2mm"}

    fig2, axes = plt.subplots(3, 3, figsize=(14, 10))
    for row, (material_key, material_density, material_thicknesses) in enumerate(mat_configs):
        for col, prob_name in enumerate(prob_names):
            ax = axes[row, col]
            prob_samples = problems.get(prob_name, [])
            material_samples = [s for s in prob_samples if s.material_key == material_key]
            if not material_samples:
                ax.set_title(f"{material_key} - {prob_labels.get(prob_name, prob_name)}\n(no data)")
                continue

            # measured data
            measured_thickness = [s.thickness_mm for s in material_samples]
            measured_ratio = [s.ratio for s in material_samples]
            ax.scatter(measured_thickness, measured_ratio, marker='o', s=50,
                       facecolors='none', edgecolors=prob_colors[prob_name],
                       linewidths=1.5, zorder=5, label='measured')

            # fitted curve using reconstructed weights
            weights = weight_map.get(prob_name)
            if weights is not None:
                d_fine = np.linspace(min(material_thicknesses), max(material_thicknesses), 100)
                A_fine = build_material_matrix(
                    basis=basis,
                    mu_mass=mu_map[f"{material_key}_mu_cm2_g"],
                    density=material_density,
                    thicknesses_mm=d_fine.tolist(),
                )
                fitted_fine = A_fine @ weights
                ax.plot(d_fine, fitted_fine, '-', color=prob_colors[prob_name],
                        linewidth=1.5, label='fitted')

            ax.set_xlabel("Thickness (mm)")
            ax.set_ylabel("I / I0")
            ax.set_yscale("log")
            ax.set_title(f"{material_key} - {prob_labels.get(prob_name, prob_name)}")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3, linestyle='--')

    fig2.suptitle(f"{kvp} kV — measured vs fitted transmission", fontsize=13, fontweight='bold')
    fig2.tight_layout()
    fig2.savefig(str(out_dir / f"fit_transmission_{kvp}kV.png"), dpi=150)
    print(f"  Saved chart: {out_dir / f'fit_transmission_{kvp}kV.png'}")

    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="基于阶梯透过率数据反演低能/高能输入能谱。")
    parser.add_argument("--kv", type=int, nargs="*", default=list(VOLTAGES), help="要处理的电压列表")
    parser.add_argument("--n-iter", type=int, default=10000, help="CMD 速度 Verlet 最大迭代次数")
    parser.add_argument("--dt", type=float, default=0.0, help="CMD 虚拟时间步长 (0=自动缩放)")
    parser.add_argument("--fit-n", type=int, default=8, help="每种材料用于拟合的厚度阶梯数 (≤10)")
    parser.add_argument(
        "--fit-mode",
        choices=["thin", "thick"],
        default="thin",
        help="拟合厚度选取模式: thin=最薄N个, thick=最厚N个",
    )
    parser.add_argument(
        "--basis-mode",
        choices=["all", "match"],
        default="all",
        help="基谱选取模式: all=所有≤kV的基谱, match=仅对应kV的基谱",
    )
    parser.add_argument(
        "--method",
        choices=["cmd", "cg"],
        default="cg",
        help="优化方法: cmd=速度Verlet, cg=投影共轭梯度",
    )
    parser.add_argument("--warmup", type=int, default=10, help="先用 CMD 预热迭代次数 (0=跳过, 所有方法通用)")
    parser.add_argument("--weight-power", type=float, default=0.1, help="样本权重幂指数: w_i ∝ thickness^(-p), p>0 薄厚度权重更大")
    args = parser.parse_args()
    fit_n = max(1, min(10, args.fit_n))

    # ── 可调参数 ──
    material_weights: Dict[str, float] = {"Cu": 0.4, "Fe": 0.4, "Al": 0.2}  # None=不按材料加权

    OUT_DIR.mkdir(exist_ok=True, parents=True)
    for kvp in args.kv:
        run_voltage(
            kvp=kvp, n_iter=args.n_iter, dt=args.dt, fit_n=fit_n,
            basis_mode=args.basis_mode, fit_mode=args.fit_mode,
            method=args.method,
            warmup=args.warmup, weight_power=args.weight_power,
            material_weights=material_weights,
        )


if __name__ == "__main__":
    main()
