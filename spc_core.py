"""
spc_core.py
Funciones core para cálculos SPC:
- constantes (A2, D3, D4, d2, B3/B4) para n=2..10 con fallback
- Xbar-R (media por subgrupo y límites)
- R chart (rangos y límites)
- I-MR (individuos + moving range)
- p-chart y c-chart básicas
- Cp / Cpk
- detection helpers (sin reglas, devuelve estadísticas básicas)
"""
import numpy as np
import pandas as pd
from typing import Dict, Any

# Tablas de constantes para n=2..10 (valores estándar)
A2_dict = {2:1.880,3:1.023,4:0.729,5:0.577,6:0.483,7:0.419,8:0.373,9:0.337,10:0.308}
D3_dict = {2:0.000,3:0.000,4:0.000,5:0.000,6:0.000,7:0.076,8:0.136,9:0.184,10:0.223}
D4_dict = {2:3.267,3:2.574,4:2.282,5:2.114,6:2.004,7:1.924,8:1.864,9:1.816,10:1.777}
d2_dict = {2:1.128,3:1.693,4:2.059,5:2.326,6:2.534,7:2.704,8:2.847,9:2.970,10:3.078}
B3_dict = {2:0.0,3:0.0,4:0.0,5:0.0,6:0.03,7:0.118,8:0.185,9:0.239,10:0.284}
B4_dict = {2:3.267,3:2.568,4:2.266,5:2.089,6:1.97,7:1.882,8:1.815,9:1.758,10:1.716}

def get_constants(n: int) -> Dict[str,float]:
    """
    Devuelve constantes A2, D3, D4, d2, B3, B4 para n.
    Si n fuera de rango, aproxima usando última disponible (n=10).
    """
    if n < 2:
        raise ValueError("n debe ser >= 2")
    if n > 10:
        # fallback simple: use n=10 constants (could implementar interpolación)
        n_use = 10
    else:
        n_use = n
    return {
        'A2': A2_dict.get(n_use),
        'D3': D3_dict.get(n_use),
        'D4': D4_dict.get(n_use),
        'd2': d2_dict.get(n_use),
        'B3': B3_dict.get(n_use),
        'B4': B4_dict.get(n_use)
    }

def xbar_r(df_wide: pd.DataFrame) -> Dict[str, Any]:
    """
    Calcula Xbar-R para df_wide (subgrupos por fila, columnas mediciones).
    Devuelve dict con medias por subgrupo, media general, R_bar, LCL/UCL para Xbar y R.
    Maneja NaN dentro de subgrupos.
    """
    # n_eff por fila (cantidad de mediciones no-NaN)
    counts = df_wide.count(axis=1)
    # medias por fila usando skipna
    medias = df_wide.mean(axis=1, skipna=True)
    media_general = medias.mean()
    # rangos por fila (max-min ignorando NaN)
    rangos = df_wide.max(axis=1, skipna=True) - df_wide.min(axis=1, skipna=True)
    r_bar = rangos.mean()

    # Atención: si n varía por fila, usamos n típico (modo) para A2; para precisión podemos usar A2 por subgrupo
    n_mode = int(counts.mode()[0]) if len(counts)>0 else df_wide.shape[1]
    const = get_constants(n_mode)
    A2 = const['A2']
    xbar_UCL = media_general + A2 * r_bar
    xbar_LCL = media_general - A2 * r_bar

    # R limits usando D3/D4
    D3 = const['D3']
    D4 = const['D4']
    R_UCL = D4 * r_bar
    R_LCL = D3 * r_bar

    return {
        'medias': medias,
        'media_general': media_general,
        'rangos': rangos,
        'r_bar': r_bar,
        'xbar_UCL': xbar_UCL,
        'xbar_LCL': xbar_LCL,
        'R_UCL': R_UCL,
        'R_LCL': R_LCL,
        'n_mode': n_mode,
        'counts': counts
    }

def imr(series: pd.Series) -> Dict[str, Any]:
    """
    Calcula chart de Individuos + Moving Range.
    series: pandas Series de observaciones individuales (en orden temporal).
    Retorna mu, MR_bar, sigma_hat, límites y MR series.
    """
    s = series.dropna().astype(float).reset_index(drop=True)
    if s.shape[0] < 2:
        raise ValueError("Se requieren al menos 2 observaciones para I-MR")
    mr = s.diff().abs().dropna()
    mr_bar = mr.mean()
    # d2 para MR de tamaño 2 es 1.128
    d2_mr = 1.128
    sigma_hat = mr_bar / d2_mr
    mu = s.mean()
    UCL = mu + 3*sigma_hat
    LCL = mu - 3*sigma_hat
    return {'mu': mu, 'mr': mr, 'mr_bar': mr_bar, 'sigma_hat': sigma_hat, 'UCL': UCL, 'LCL': LCL}

def p_chart(defects: np.ndarray, n: np.ndarray) -> Dict[str, Any]:
    """
    p-chart: defects = array of defect counts; n = denominators (opportunities) per subgroup.
    Retorna p, p_bar, límites por subgrupo.
    """
    defects = np.asarray(defects, dtype=float)
    n = np.asarray(n, dtype=float)
    p = defects / n
    p_bar = defects.sum() / n.sum()
    sigma = np.sqrt(p_bar*(1-p_bar)/n)
    UCL = p_bar + 3*sigma
    LCL = p_bar - 3*sigma
    LCL = np.maximum(LCL, 0.0)
    return {'p': p, 'p_bar': p_bar, 'UCL': UCL, 'LCL': LCL, 'sigma': sigma}

def c_chart(counts: np.ndarray) -> Dict[str, Any]:
    """
    c-chart: counts por subgrupo (Poisson). Límite basado en sqrt(lambda).
    """
    counts = np.asarray(counts, dtype=float)
    c_bar = counts.mean()
    sigma = np.sqrt(c_bar)
    UCL = c_bar + 3*sigma
    LCL = c_bar - 3*sigma
    LCL = np.maximum(LCL, 0.0)
    return {'counts': counts, 'c_bar': c_bar, 'UCL': UCL, 'LCL': LCL}

def capability(values: np.ndarray, LSL: float, USL: float, ddof: int = 1) -> Dict[str, Any]:
    """
    Calcula Cp y Cpk en base a valores numéricos (ignorando NaN).
    ddof: 1 por defecto (muestra), se puede usar 0 para población si se quiere.
    """
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return {'Cp': np.nan, 'Cpk': np.nan, 'Media': np.nan, 'Std': np.nan}
    mu = vals.mean()
    sigma = vals.std(ddof=ddof)
    Cp = (USL - LSL) / (6*sigma) if sigma > 0 else np.nan
    Cpk = min((USL - mu) / (3*sigma), (mu - LSL) / (3*sigma)) if sigma > 0 else np.nan
    return {'Cp': Cp, 'Cpk': Cpk, 'Media': mu, 'Std': sigma}