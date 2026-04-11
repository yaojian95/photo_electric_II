import argparse
import math
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def _load_xyz(path: Path):
    """Load scintillator voxel list.

    Format (after header lines):
      col2-4: x,y,z (mm)
      col5: EnergyDeposit (MeV)
      col6: Norm255 (0-255)

    Returns numpy arrays (x, y, z, edep, norm255).
    """
    xs = []
    ys = []
    zs = []
    edeps = []
    norms = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                continue
            # first line may be a single integer (count)
            parts = s.split()
            if len(parts) == 1:
                # e.g. "592500"
                continue
            if len(parts) < 6:
                continue

            try:
                # type = float(parts[0])  # unused
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                edep = float(parts[4])
                norm = float(parts[5])
            except ValueError:
                continue

            xs.append(x)
            ys.append(y)
            zs.append(z)
            edeps.append(edep)
            norms.append(norm)

    return (
        np.asarray(xs, dtype=float),
        np.asarray(ys, dtype=float),
        np.asarray(zs, dtype=float),
        np.asarray(edeps, dtype=float),
        np.asarray(norms, dtype=float),
    )


def _grid_xy(xs: np.ndarray, zs: np.ndarray, values: np.ndarray, *, dx: float, dz: float):
    """Bin scattered x/z points to a regular grid using voxel size dx/dz (mm)."""
    x0 = float(xs.min())
    z0 = float(zs.min())
    nx = int(math.floor((float(xs.max()) - x0) / dx + 0.5)) + 1
    nz = int(math.floor((float(zs.max()) - z0) / dz + 0.5)) + 1

    # indices via rounding to nearest voxel center
    ix = np.rint((xs - x0) / dx).astype(int)
    iz = np.rint((zs - z0) / dz).astype(int)

    grid = np.full((nz, nx), np.nan, dtype=float)

    # If there are duplicates, accumulate (sum) and keep count for averaging.
    acc = np.zeros((nz, nx), dtype=float)
    cnt = np.zeros((nz, nx), dtype=int)

    valid = (ix >= 0) & (ix < nx) & (iz >= 0) & (iz < nz)
    ix = ix[valid]
    iz = iz[valid]
    vv = values[valid]

    np.add.at(acc, (iz, ix), vv)
    np.add.at(cnt, (iz, ix), 1)

    m = cnt > 0
    grid[m] = acc[m] / cnt[m]

    extent = (x0 - dx / 2.0, x0 + (nx - 1) * dx + dx / 2.0, z0 - dz / 2.0, z0 + (nz - 1) * dz + dz / 2.0)
    return grid, extent


def plot_xz_heatmap(
    xyz_path: str,
    *,
    dx: float = 0.8,
    dz: float = 0.72,
    value_column: str = "norm255",
    cmap: str = "inferno",
    vmin=None,
    vmax=None,
    output: Optional[str] = None,
    show: bool = True,
):
    """Plot x-z plane heatmap. 支持0~65535或0~255归一化。"""
    p = Path(xyz_path)
    xs, ys, zs, edep, norm = _load_xyz(p)

    # 判断是否16位输出
    use_16bit = (dx == 0.4 and dz == 0.4)
    out_max = 65535 if use_16bit else 255

    if value_column.lower() in {"norm", "norm255", "gray", "pixel"}:
        values = norm
        title = f"{p.name}  (Norm255)"
        if vmin is None:
            vmin = 0
        if vmax is None:
            vmax = out_max
    elif value_column.lower() in {"edep", "energy", "energdeposit", "mev"}:
        values = edep
        title = f"{p.name}  (EnergyDeposit MeV)"
    else:
        raise ValueError("value_column must be 'norm255' or 'edep'")

    grid_xz, extent_xz = _grid_xy(xs, zs, values, dx=dx, dz=dz)

    # Swap axes: make z horizontal and x vertical.
    
    x_min, x_max, z_min, z_max = extent_xz
    extent_zx = (z_min, z_max, x_min, x_max)

    z_span = max(1e-9, z_max - z_min)
    x_span = max(1e-9, x_max - x_min)

    # 归一化到out_max
    grid_zx = grid_xz.T  # (nx, nz)
    img = grid_zx[1:-1, 1:-1]
    # 背景展开矫正
    # 获取z坐标范围
    # 1690.801 37.40006
    z0 = float(np.min(xs)) - 0.5 * dx
    dz_val = dz
    # 区间定义（单位mm）
    z_ranges = [(2086, 2094), (1691, 1699)] #(1880, 1900), (1980, 2013), (1781, 1800), 
    bg_rows = []
    for zmin, zmax in z_ranges:
        # 找到对应的行索引
        idx_min = int(round((zmin - z0) / dz_val)) - 1  # -1因img已去边
        idx_max = int(round((zmax - z0) / dz_val)) - 1
        idx_min = max(idx_min, 0)
        idx_max = min(idx_max, img.shape[0]-1)
        if idx_max < idx_min:
            continue
        rows = img[idx_min:idx_max+1, :]
        if rows.size > 0:
            bg_rows.append(np.nanmean(rows, axis=0))
    if bg_rows:
        bg_profile = np.nanmean(np.stack(bg_rows, axis=0), axis=0)  # 1D背景行
        mean_bg = np.nanmean(bg_profile)
        coef_profile = bg_profile / (mean_bg if mean_bg != 0 else 1e-6)  # 校正系数行
        coef_profile[coef_profile==0] = 1e-6  # 防止除零
        bg_matrix = np.tile(coef_profile, (img.shape[0], 1))  # 扩展为背景校正矩阵
        img = img / bg_matrix

    img_min, img_max = np.nanmin(img), np.nanmax(img)
    img = (img - img_min) / (img_max - img_min + 1e-9) * out_max
    img = np.nan_to_num(img, nan=0.0)
    img = np.round(img)
    img = img.astype(np.uint16 if use_16bit else np.uint8)

    # Create a wide canvas and use GridSpec so the image occupies most space.
    fig_w = 14.0
    fig_h = max(4.5, min(7.5, fig_w * (x_span / z_span)))

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[0.8, 0.015], wspace=0.04)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])

    im = ax.imshow(
        img,
        origin="lower",
        extent=extent_zx,
        aspect="equal",
        interpolation="nearest",
        cmap=cmap,
        vmin=0,
        vmax=out_max,
    )

    ax.set_xlabel("z (mm)")
    ax.set_ylabel("x (mm)")
    ax.set_title("")  # no title

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    cbar = fig.colorbar(im, cax=cax)
    if use_16bit:
        ticks = np.linspace(0, 65535, 6, dtype=int)
        cbar.set_ticks(ticks)
        cbar.ax.set_yticklabels([str(t) for t in ticks])
    else:
        cbar.set_ticks(np.array([0, 50, 100, 200, 255]))
        cbar.ax.set_yticklabels(["0", "50", "100", "200", "255"])
    cbar.set_label("")

    fig.subplots_adjust(top=0.935,
                        bottom=0.125,
                        left=0.055,
                        right=0.92,
                        hspace=0.11,
                        wspace=0.035)

    if output:
        outp = Path(output)
        outp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outp, dpi=300, bbox_inches="tight", pad_inches=0.01)

    if show:
        plt.show()
    plt.close(fig)


def _find_xyz_pair(data_dir: str) -> Optional[Tuple[str, str]]:
    """查找目录下最新编号的scintillator_Gd2O2S-1/2_sp_N.xyz或_edep_N.xyz成对文件。"""
    import re
    import os
    files = [f for f in os.listdir(data_dir) if f.endswith('.xyz')]
    pairs = {}
    for fname in files:
        m1 = re.match(r"scintillator_Gd2O2S-1_(sp|edep)_(\d+)\.xyz$", fname)
        m2 = re.match(r"scintillator_Gd2O2S-2_(sp|edep)_(\d+)\.xyz$", fname)
        if m1:
            n = int(m1.group(2))
            key = m1.group(1)  # 'sp' or 'edep'
            pairs.setdefault((key, n), [None, None])[0] = os.path.join(data_dir, fname)
        if m2:
            n = int(m2.group(2))
            key = m2.group(1)
            pairs.setdefault((key, n), [None, None])[1] = os.path.join(data_dir, fname)
    valid = [((key, n), v) for (key, n), v in pairs.items() if v[0] and v[1]]
    if valid:
        # 优先sp，其次edep，编号最大
        valid.sort(key=lambda x: (0 if x[0][0]=="sp" else 1, x[0][1]), reverse=True)
        (_, _), (low, high) = valid[0]
        return low, high
    return None


def main():
    ap = argparse.ArgumentParser(description="Plot x-z heatmap from scintillator voxel XYZ file.")
    ap.add_argument(
        "input_pos",
        nargs="?",
        default=None,
        help="Path to *.xyz file or directory (positional, optional). If omitted, --input or the default path is used.",
    )
    ap.add_argument(
        "--input",
        default=None,
        help="Path to *.xyz file or directory (optional if positional <file> is provided)",
    )
    ap.add_argument("--dx", type=float, default=0.4, help="Voxel size along x in mm")
    ap.add_argument("--dz", type=float, default=0.4, help="Voxel size along z in mm")
    ap.add_argument("--value", choices=["norm255", "edep"], default="norm255", help="Which column to plot")
    ap.add_argument("--cmap", default="gray")
    ap.add_argument("--vmin", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--output", default=None, help="If set, save figure to this path")
    ap.add_argument("--no-show", action="store_true", help="Do not open a window")
    args = ap.parse_args()

    input_path = args.input_pos or args.input

    if input_path and os.path.isdir(input_path):
        # 目录模式，自动查找高低能xyz
        pair = _find_xyz_pair(input_path)
        if not pair:
            print(f"No valid scintillator_Gd2O2S-1/2_sp_N.xyz or _edep_N.xyz pair found in {input_path}")
            return
        low_xyz, high_xyz = pair
        print(f"Found latest xyz pair: {low_xyz}, {high_xyz}")
        # 上下显示两幅图，每幅图有独立colorbar，整体更紧凑
        fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        for ax, xyz, title in zip(axes, [low_xyz, high_xyz], ["Gd2O2S-1", "Gd2O2S-2"]):
            xs, ys, zs, edep, norm = _load_xyz(Path(xyz))
            grid_edep, extent_xz = _grid_xy(xs, zs, edep, dx=args.dx, dz=args.dz)
            grid_zx_edep = grid_edep.T
            # 去掉最外一圈像素
            img = grid_zx_edep[1:-1, 1:-1]
            # 背景展开矫正
            # 获取z坐标范围
            # 1690.801 37.40006
            z0 = 1690.801
            dz_val = args.dz
            # 区间定义（单位mm）
            z_ranges = [(2086, 2094), (1691, 1699), (1980, 2013), (1781, 1800)] #(1880, 1900), (1980, 2013), (1781, 1800), 
            bg_rows = []
            for zmin, zmax in z_ranges:
                # 找到对应的行索引
                idx_min = int(round((zmin - z0) / dz_val)) - 1  # -1因img已去边
                idx_max = int(round((zmax - z0) / dz_val)) - 1
                idx_min = max(idx_min, 0)
                idx_max = min(idx_max, img.shape[0]-1)
                if idx_max < idx_min:
                    continue
                rows = img[idx_min:idx_max+1, :]
                if rows.size > 0:
                    bg_rows.append(np.nanmean(rows, axis=0))
            if bg_rows:
                bg_profile = np.nanmean(np.stack(bg_rows, axis=0), axis=0)  # 1D背景行
                mean_bg = np.nanmean(bg_profile)
                coef_profile = bg_profile / (mean_bg if mean_bg != 0 else 1e-6)  # 校正系数行
                coef_profile[coef_profile==0] = 1e-6  # 防止除零
                bg_matrix = np.tile(coef_profile, (img.shape[0], 1))  # 扩展为背景校正矩阵
                img = img / bg_matrix
     
            edep_min, edep_max = np.nanmin(img), np.nanmax(img)
            use_16bit = False#(args.dx == 0.4 and args.dz == 0.4)
            out_max = 65535 if use_16bit else 255
            img = (img - edep_min) / (edep_max - edep_min + 1e-9) * out_max
            img = np.nan_to_num(img, nan=0.0)
            img = np.round(img)
            img = img.astype(np.uint16 if use_16bit else np.uint8)
            x_min, x_max, z_min, z_max = extent_xz
            nx, nz = grid_zx_edep.shape
            extent_zx = (
                z_min + (z_max - z_min) / nz,
                z_max - (z_max - z_min) / nz,
                x_min + (x_max - x_min) / nx,
                x_max - (x_max - x_min) / nx,
            )
            im = ax.imshow(
                img,
                origin="lower",
                extent=extent_zx,
                aspect="equal",
                interpolation="nearest",
                cmap=args.cmap,
                vmin=0,
                vmax=out_max,
            )
            ax.set_title(title)
            ax.set_ylabel("x (mm)")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.06)
            cbar = fig.colorbar(im, cax=cax, orientation='vertical')
            if use_16bit:
                ticks = np.linspace(0, 65535, 6, dtype=int)
                cbar.set_ticks(ticks)
                cbar.set_ticklabels([str(t) for t in ticks])
                cbar.set_label('Gray value (0-65535)')
            else:
                cbar.set_label('Gray value (0-255)')
                cbar.set_ticks([0, 50, 100, 150, 200, 255])
                cbar.set_ticklabels(["0", "50", "100", "150","200", "255"])
        axes[-1].set_xlabel("z (mm)")
        plt.tight_layout(pad=1.0, h_pad=0.275)
        fig.subplots_adjust(top=0.995,
                            bottom=0.09,
                            left=0.093,
                            right=0.898,
                            hspace=0.0,
                            wspace=0.155)
        if not args.no_show:
            plt.show()
        plt.close(fig)
        return

    plot_xz_heatmap(
        input_path,
        dx=args.dx,
        dz=args.dz,
        value_column=args.value,
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
        output=args.output,
        show=(not args.no_show),
    )


if __name__ == "__main__":
    main()
