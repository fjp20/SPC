"""
rr_core.py
Funciones para R&R por atributos:
- resumen por operador vs referencia (porcentaje acuerdo, cohen kappa)
- matriz Kappa entre operadores
- matriz de confusión y heatmap (exportable desde la app)
"""
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

def summary_vs_reference(df: pd.DataFrame,
                         operator_col: str = "Operador",
                         reference_col: str = "Referencia",
                         eval_col: str = "Evaluación") -> pd.DataFrame:
    """
    Para cada operador calcula %acuerdo con referencia y Kappa contra referencia.
    Devuelve DataFrame resumen.
    """
    operadores = df[operator_col].unique()
    rows = []
    for op in operadores:
        sub = df[df[operator_col] == op]
        # Aseguramos orden comparable si hay piezas repetidas
        agree_pct = np.mean(sub[eval_col] == sub[reference_col]) * 100
        kappa = cohen_kappa_score(sub[eval_col], sub[reference_col])
        rows.append({operator_col: op, "% Acuerdo con referencia": round(agree_pct,2), "Kappa": round(kappa,3)})
    return pd.DataFrame(rows)

def kappa_matrix_between_operators(df: pd.DataFrame,
                                   operator_col: str = "Operador",
                                   piece_col: str = "Pieza",
                                   eval_col: str = "Evaluación") -> pd.DataFrame:
    """
    Calcula matriz de Kappa entre operadores usando piezas comunes.
    """
    ops = list(df[operator_col].unique())
    mat = pd.DataFrame(index=ops, columns=ops, dtype=float)
    for i in ops:
        for j in ops:
            if i == j:
                mat.loc[i,j] = 1.0
            else:
                piezas_i = set(df[df[operator_col]==i][piece_col])
                piezas_j = set(df[df[operator_col]==j][piece_col])
                common = piezas_i & piezas_j
                if not common:
                    mat.loc[i,j] = np.nan
                    continue
                di = df[(df[operator_col]==i) & (df[piece_col].isin(common))].sort_values(piece_col)[eval_col]
                dj = df[(df[operator_col]==j) & (df[piece_col].isin(common))].sort_values(piece_col)[eval_col]
                mat.loc[i,j] = cohen_kappa_score(di, dj)
    return mat

def confusion_matrix_for_operator(df: pd.DataFrame,
                                  operator: str,
                                  operator_col: str = "Operador",
                                  piece_col: str = "Pieza",
                                  eval_col: str = "Evaluación",
                                  reference_col: str = "Referencia") -> pd.DataFrame:
    """
    Devuelve matriz de confusión (DataFrame) del operador contra referencia (clasificaciones).
    """
    sub = df[df[operator_col] == operator]
    if sub.empty:
        return pd.DataFrame()
    y_true = sub[reference_col]
    y_pred = sub[eval_col]
    labels = np.unique(np.concatenate([y_true.unique(), y_pred.unique()]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=labels, columns=labels)