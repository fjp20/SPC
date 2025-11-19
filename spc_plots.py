"""
spc_plots.py
Funciones para graficar charts SPC usando matplotlib y opcionalmente plotly para interactividad.
Incluye marcado de puntos fuera de control (simple: fuera de LCL/UCL) y función para dibujar Xbar+R juntos.
"""
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import pandas as pd

def plot_xbar_r_matplotlib(res: dict):
    """
    Dibuja Xbar y R con matplotlib. 'res' viene de spc_core.xbar_r()
    Retorna figura matplotlib.
    """
    medias = res['medias']
    rangos = res['rangos']
    fig, axes = plt.subplots(2,1, figsize=(10,8), sharex=True)
    ax1, ax2 = axes
    # Xbar
    ax1.plot(medias.index, medias.values, marker='o', linestyle='-')
    ax1.axhline(res['media_general'], color='green', label=f"Center {res['media_general']:.3f}")
    ax1.axhline(res['xbar_UCL'], color='red', linestyle='--', label=f"UCL {res['xbar_UCL']:.3f}")
    ax1.axhline(res['xbar_LCL'], color='red', linestyle='--', label=f"LCL {res['xbar_LCL']:.3f}")
    # marcar fuera
    fuera_x = (medias > res['xbar_UCL']) | (medias < res['xbar_LCL'])
    if fuera_x.any():
        ax1.plot(medias.index[fuera_x], medias.values[fuera_x], 'rx', markersize=10)
    ax1.set_ylabel("Media")
    ax1.legend()

    # R
    ax2.plot(rangos.index, rangos.values, marker='o', linestyle='-')
    ax2.axhline(res['r_bar'], color='green', label=f"R̄ {res['r_bar']:.3f}")
    ax2.axhline(res['R_UCL'], color='red', linestyle='--', label=f"UCL {res['R_UCL']:.3f}")
    ax2.axhline(res['R_LCL'], color='red', linestyle='--', label=f"LCL {res['R_LCL']:.3f}")
    fuera_r = (rangos > res['R_UCL']) | (rangos < res['R_LCL'])
    if fuera_r.any():
        ax2.plot(rangos.index[fuera_r], rangos.values[fuera_r], 'rx', markersize=10)
    ax2.set_ylabel("Rango")
    ax2.set_xlabel("Subgrupo")
    ax2.legend()
    plt.tight_layout()
    return fig

def plot_xbar_plotly(res: dict):
    """
    Versión interactiva con plotly que devuelve objeto Figure.
    """
    medias = res['medias']
    rangos = res['rangos']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=medias.index, y=medias.values, mode='lines+markers', name='Medias'))
    fig.add_hline(y=res['media_general'], line=dict(color='green'), annotation_text="Center")
    fig.add_hline(y=res['xbar_UCL'], line=dict(color='red', dash='dash'), annotation_text="UCL")
    fig.add_hline(y=res['xbar_LCL'], line=dict(color='red', dash='dash'), annotation_text="LCL")
    # R chart como subplot simple: lo presentaremos como segunda figura si se desea
    return fig