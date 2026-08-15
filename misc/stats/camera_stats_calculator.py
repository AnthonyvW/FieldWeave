from dataclasses import dataclass
from typing import Iterable, Tuple, List
import math

# ---------- Core optics math ----------

def nyquist_object_pixel_size_um(na: float, wavelength_um: float = 0.55) -> float:
    if na <= 0 or wavelength_um <= 0:
        raise ValueError("NA and wavelength_um must be > 0.")
    return wavelength_um / (4.0 * na)

def object_pixel_size_um(sensor_pixel_um: float, magnification: float) -> float:
    if sensor_pixel_um <= 0 or magnification <= 0:
        raise ValueError("sensor_pixel_um and magnification must be > 0.")
    return sensor_pixel_um / magnification

def oversampling_factor(na: float, magnification: float, sensor_pixel_um: float, wavelength_um: float = 0.55) -> float:
    p_nyq = nyquist_object_pixel_size_um(na, wavelength_um)
    p_obj = object_pixel_size_um(sensor_pixel_um, magnification)
    return p_nyq / p_obj

def fov_um(nx: int, ny: int, sensor_pixel_um: float, magnification: float) -> Tuple[float, float]:
    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be > 0.")
    p_obj = object_pixel_size_um(sensor_pixel_um, magnification)
    return (nx * p_obj, ny * p_obj)

def dpi_from_fov(nx: int, ny: int, sensor_pixel_um: float, magnification: float) -> float:
    """Compute dots per inch (DPI) in object space for the base (no binning) image."""
    fov_x_um, _ = fov_um(nx, ny, sensor_pixel_um, magnification)
    inches_per_um = 1.0 / 25_400.0  # µm to inches
    return nx / (fov_x_um * inches_per_um)

# ---------- Diffraction & MTF helpers ----------

def airy_diameter_obj_um(na: float, wavelength_um: float = 0.55) -> float:
    return 1.22 * wavelength_um / na

def optical_cutoff_obj_cyc_per_um(na: float, wavelength_um: float = 0.55) -> float:
    return (2.0 * na) / wavelength_um

def pixel_nyquist_obj_cyc_per_um(sensor_pixel_um: float, magnification: float) -> float:
    p_obj = sensor_pixel_um / magnification
    return 1.0 / (2.0 * p_obj)

def mtf_diffraction_incoherent(fc_cyc_per_um: float, f_cyc_per_um: float) -> float:
    if f_cyc_per_um >= fc_cyc_per_um:
        return 0.0
    nu = f_cyc_per_um / fc_cyc_per_um
    return (2.0 / math.pi) * (math.acos(nu) - nu * math.sqrt(1.0 - nu * nu))

def pixels_per_airy_diameter(na: float, magnification: float, sensor_pixel_um: float, wavelength_um: float = 0.55) -> float:
    D_airy = airy_diameter_obj_um(na, wavelength_um)
    p_obj = sensor_pixel_um / magnification
    return D_airy / p_obj

# ---------- Binning selection ----------

def ideal_binning_for_autofocus(
    na: float,
    magnification: float,
    sensor_pixel_um: float,
    wavelength_um: float = 0.55,
    available_binnings: Iterable[int] = (1, 2, 3, 4, 8),
    mode: str = "ceil",
) -> int:
    """
    Integer bin based on oversampling S. 'ceil' removes oversampling.
    """
    S = oversampling_factor(na, magnification, sensor_pixel_um, wavelength_um)
    bins = sorted(set(int(b) for b in available_binnings if b >= 1))
    if not bins:
        raise ValueError("available_binnings must contain at least one positive integer.")
    if mode == "ceil":
        for b in bins:
            if b >= S:
                return b
        return bins[-1]
    if mode == "floor":
        le = [b for b in bins if b <= S]
        return le[-1] if le else bins[0]
    # nearest
    return min(bins, key=lambda b: abs(b - S))

# ---------- Internal: compute metrics for a (possibly fractional) bin ----------

@dataclass(frozen=True)
class BinnedMetrics:
    bin_factor: float
    nx: int
    ny: int
    mpix: float
    p_obj_um: float
    pixels_per_airy_diam: float
    pixel_nyquist_cyc_per_um: float
    fN_to_fc_ratio: float
    mtf_at_pixel_nyquist: float
    oversampling_S: float
    dpi: float       

def _compute_binned_metrics(
    bin_factor: float,
    nx_base: int,
    ny_base: int,
    na: float,
    magnification: float,
    sensor_pixel_um: float,
    wavelength_um: float,
) -> BinnedMetrics:
    if bin_factor <= 0:
        raise ValueError("bin_factor must be > 0.")
    # Effective pixel & object pixel
    p_sensor_eff = bin_factor * sensor_pixel_um
    p_obj_eff = p_sensor_eff / magnification

    # Resulting resolution (rounded for fractional)
    nx = max(1, int(round(nx_base / bin_factor)))
    ny = max(1, int(round(ny_base / bin_factor)))
    mpix = round((nx * ny) / 1_000_000.0, 3)
    fov_x_um, _ = fov_um(nx, ny, bin_factor * sensor_pixel_um, magnification)
    dpi = nx / (fov_x_um / 25_400.0)

    # Frequency-domain + diffraction context
    fc = optical_cutoff_obj_cyc_per_um(na, wavelength_um)
    fN = 1.0 / (2.0 * p_obj_eff)
    fN_to_fc = fN / fc if fc > 0 else float("inf")
    mtfN = mtf_diffraction_incoherent(fc, fN)
    D_airy = airy_diameter_obj_um(na, wavelength_um)
    px_per_airy = D_airy / p_obj_eff
    p_nyq_obj = nyquist_object_pixel_size_um(na, wavelength_um)
    S_after = p_nyq_obj / p_obj_eff

    return BinnedMetrics(
        bin_factor=bin_factor,
        nx=nx,
        ny=ny,
        mpix=mpix,
        p_obj_um=p_obj_eff,
        pixels_per_airy_diam=px_per_airy,
        pixel_nyquist_cyc_per_um=fN,
        fN_to_fc_ratio=fN_to_fc,
        mtf_at_pixel_nyquist=mtfN,
        oversampling_S=S_after,
        dpi=dpi,
    )

# ---------- Sampling Report (inputs + pre-binning) ----------

@dataclass(frozen=True)
class SamplingReport:
    title: str
    nx: int
    ny: int
    p_nyq_obj_um: float
    p_obj_um: float
    oversampling_S: float
    fov_x_um: float
    fov_y_um: float
    airy_diameter_um: float
    optical_cutoff_cyc_per_um: float
    pixel_nyquist_cyc_per_um: float
    mtf_at_pixel_nyquist: float
    pixels_per_airy_diam: float
    suggested_binning: int   # ceil(S)
    dpi: float               # base DPI in object space

def sampling_summary(
    title: str,
    nx: int,
    ny: int,
    sensor_pixel_um: float,
    magnification: float,
    na: float,
    wavelength_um: float = 0.55,
    available_binnings: Iterable[int] = (1, 2, 3, 4, 8),
    mode: str = "ceil",
) -> SamplingReport:
    p_nyq = nyquist_object_pixel_size_um(na, wavelength_um)
    p_obj = object_pixel_size_um(sensor_pixel_um, magnification)
    S = p_nyq / p_obj
    fovx, fovy = fov_um(nx, ny, sensor_pixel_um, magnification)
    D_airy = airy_diameter_obj_um(na, wavelength_um)
    fc = optical_cutoff_obj_cyc_per_um(na, wavelength_um)
    fN = pixel_nyquist_obj_cyc_per_um(sensor_pixel_um, magnification)
    mtfN = mtf_diffraction_incoherent(fc, fN)
    px_per_airy = D_airy / p_obj
    dpi = dpi_from_fov(nx, ny, sensor_pixel_um, magnification)

    b_ideal_rounded = ideal_binning_for_autofocus(
        na, magnification, sensor_pixel_um, wavelength_um, available_binnings, mode
    )

    return SamplingReport(
        title=title,
        nx=nx, ny=ny,
        p_nyq_obj_um=p_nyq,
        p_obj_um=p_obj,
        oversampling_S=S,
        fov_x_um=fovx,
        fov_y_um=fovy,
        airy_diameter_um=D_airy,
        optical_cutoff_cyc_per_um=fc,
        pixel_nyquist_cyc_per_um=fN,
        mtf_at_pixel_nyquist=mtfN,
        pixels_per_airy_diam=px_per_airy,
        suggested_binning=b_ideal_rounded,
        dpi=dpi,
    )

# ---------- Per-camera 3-column table (Base | Ideal (S) | Ideal ⌈rounded⌉) ----------

def render_camera_table(
    report: SamplingReport,
    sensor_pixel_um: float,
    magnification: float,
    na: float,
    wavelength_um: float = 0.55,
) -> str:
    base       = _compute_binned_metrics(1.0, report.nx, report.ny, na, magnification, sensor_pixel_um, wavelength_um)
    ideal_flt  = _compute_binned_metrics(report.oversampling_S, report.nx, report.ny, na, magnification, sensor_pixel_um, wavelength_um)
    ideal_ceil = _compute_binned_metrics(float(report.suggested_binning), report.nx, report.ny, na, magnification, sensor_pixel_um, wavelength_um)

    # If Ideal S < 1, indicate undersampling instead of a fake fractional upsampled resolution
    ideal_res_label = "Undersampling" if report.oversampling_S < 1.0 else f"~{ideal_flt.nx}×{ideal_flt.ny}"

    headers = ["Metric", "Base", "Ideal (S)", f"Ideal ⌈{report.suggested_binning}×⌉"]
    rows = [
        ["Megapixels", f"{(report.nx*report.ny)/1_000_000:.3f} MP", f"~{ideal_flt.mpix:.3f} MP", f"{ideal_ceil.mpix:.3f} MP"],
        # Show DPI for all three columns (fix)
        ["DPI", f"{base.dpi:.0f}", f"{ideal_flt.dpi:.0f}", f"{ideal_ceil.dpi:.0f}"],
        ["Bin factor", f"{base.bin_factor:.2f}", f"{ideal_flt.bin_factor:.2f}×", f"{ideal_ceil.bin_factor:.0f}×"],
        ["Result res (px)", f"{base.nx}×{base.ny}", ideal_res_label, f"{ideal_ceil.nx}×{ideal_ceil.ny}"],
        ["Obj px (µm)", f"{base.p_obj_um:.3f}", f"{ideal_flt.p_obj_um:.3f}", f"{ideal_ceil.p_obj_um:.3f}"],
        ["px / Airy", f"{base.pixels_per_airy_diam:.2f}", f"{ideal_flt.pixels_per_airy_diam:.2f}", f"{ideal_ceil.pixels_per_airy_diam:.2f}"],
        ["fN / fc", f"{base.fN_to_fc_ratio:.2f}", f"{ideal_flt.fN_to_fc_ratio:.2f}", f"{ideal_ceil.fN_to_fc_ratio:.2f}"],
        ["MTF@Nyq", f"{base.mtf_at_pixel_nyquist:.3f}", f"{ideal_flt.mtf_at_pixel_nyquist:.3f}", f"{ideal_ceil.mtf_at_pixel_nyquist:.3f}"],
        ["Oversampling S", f"{base.oversampling_S:.2f}×", f"{ideal_flt.oversampling_S:.2f}×", f"{ideal_ceil.oversampling_S:.2f}×"],
    ]

    # format pretty
    col_w = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    def fmt_row(cells: List[str]) -> str:
        return " | ".join(cell.ljust(col_w[i]) for i, cell in enumerate(cells))
    sep = "-+-".join("-"*w for w in col_w)

    out = []
    out.append(f"{report.title}  ({report.nx}×{report.ny} px)   FOV {report.fov_x_um/1000:.3f}×{report.fov_y_um/1000:.3f} mm")
    out.append(fmt_row(headers))
    out.append(sep)
    for r in rows:
        out.append(fmt_row(r))
    out.append("")  # trailing newline
    return "\n".join(out)

# ---------- Summary table (FOV, Resolution, Binning, Obj px, DPI) ----------

def render_summary_table(
    items: List[Tuple[SamplingReport, float, float, float, float]],
) -> str:
    """
    items: list of tuples (report, sensor_pixel_um, magnification, na, wavelength_um)

    Columns:
      Camera | FOV (mm) | Base Res (px) | Obj px (µm) | DPI (Base) | DPI (Ideal S) |
      Ideal S | Result Res (Ideal S) | Ideal ⌈bin⌉ | Result Res (Ideal ⌈bin⌉)
    """
    headers = [
        "Camera",
        "FOV (mm)",
        "Base Res (px)",
        "Obj px (µm)",
        "DPI (Base)",
        "DPI (Ideal S)",
        "Ideal S",
        "Result Res (Ideal S)",
        "Ideal ⌈bin⌉",
        "Result Res (Ideal ⌈bin⌉)",
    ]

    rows: List[List[str]] = []

    for rep, p_um, mag, na, lam in items:
        # Compute metrics for ideal (fractional) and ceiling binning
        ideal_S_metrics = _compute_binned_metrics(
            rep.oversampling_S, rep.nx, rep.ny, na, mag, p_um, lam
        )
        ideal_ceil_metrics = _compute_binned_metrics(
            float(rep.suggested_binning), rep.nx, rep.ny, na, mag, p_um, lam
        )

        # Label undersampling case
        ideal_S_res_label = (
            "Undersampling"
            if rep.oversampling_S < 1.0
            else f"~{ideal_S_metrics.nx}×{ideal_S_metrics.ny}"
        )

        rows.append([
            rep.title,
            f"{rep.fov_x_um/1000:.3f}×{rep.fov_y_um/1000:.3f}",
            f"{rep.nx}×{rep.ny}",
            f"{rep.p_obj_um:.3f}",
            f"{rep.dpi:,.0f}",                  # Base DPI
            f"{ideal_S_metrics.dpi:,.0f}",       # (NEW) Ideal S DPI
            f"{rep.oversampling_S:.2f}×",
            ideal_S_res_label,
            f"{rep.suggested_binning}×",
            f"{ideal_ceil_metrics.nx}×{ideal_ceil_metrics.ny}",
        ])

    # Pretty formatting
    col_w = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_w[i] = max(col_w[i], len(cell))

    def fmt_row(cells: List[str]) -> str:
        return " | ".join(cell.ljust(col_w[i]) for i, cell in enumerate(cells))

    sep = "-+-".join("-" * w for w in col_w)
    out_lines = [fmt_row(headers), sep]
    for row in rows:
        out_lines.append(fmt_row(row))
    return "\n".join(out_lines)


# ---------- Example ----------
if __name__ == "__main__":
    # You supplied mag and numerical_aperature externally.
    # Example: set them here, or import from your config.
    mag = 40
    numerical_aperature = 0.65  # NA
    lam = 0.55                  # µm
    available_binnings = (1, 2, 3, 4, 5, 6, 7, 8)  # adjust to SDK

    cameras = [
        ("MU 1000-HS",                    3584, 2748, 3.45,  mag, numerical_aperature),
        ("MU 500",                        2592, 1944, 2.20,  mag, numerical_aperature),
        ("Mokose 4k USB Webcam",          3840, 2160, 2.00,  mag, numerical_aperature),
        ("Raspberry Pi HQ Camera",        4056, 3040, 1.55,  mag, numerical_aperature),
        ("Raspberry Pi GS Camera",        1456, 1088, 3.45,  mag, numerical_aperature),
        ("Arducam 120fps camera",         5496, 3672, 2.4,  mag, numerical_aperature),
        ("Fujifilm X-M5",                 9504, 6336, 3.76,  mag, numerical_aperature),
        ("TCD1304",                       3648, 1, 8,  mag, numerical_aperature),
    ]

    reports: List[SamplingReport] = []
    items_for_summary: List[Tuple[SamplingReport, float, float, float, float]] = []

    for title, nx, ny, px_um, mag_i, na_i in cameras:
        rep = sampling_summary(
            title=title, nx=nx, ny=ny,
            sensor_pixel_um=px_um, magnification=mag_i, na=na_i,
            wavelength_um=lam,
            available_binnings=available_binnings,
            mode="ceil",
        )
        reports.append(rep)
        items_for_summary.append((rep, px_um, mag_i, na_i, lam))

    # Per-camera 3-column tables
    for (title, nx, ny, px_um, mag_i, na_i), rep in zip(cameras, reports):
        print(render_camera_table(rep, sensor_pixel_um=px_um, magnification=mag_i, na=na_i, wavelength_um=lam))
    # Summary comparison
    print(render_summary_table(items_for_summary))
