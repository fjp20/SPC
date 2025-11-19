"""
app.py
Aplicación Streamlit principal para SPC y R&R por atributos.
- Usa read_and_normalize(uploaded, auto_fix) del módulo clean_io (o spc_io con la función equivalente)
- Usa spc_core, spc_plots, rr_core para cálculos y gráficos
- Provee modo automático (limpia y sigue) y modo seguro (muestra problemas antes de analizar)
"""
import streamlit as st
import io
import pandas as pd

# Importa las utilidades esperadas. Asegúrate de tener estos archivos en el mismo directorio:
# - clean_io.py (contiene read_and_normalize)
# - spc_core.py (xbar_r, imr, p_chart, c_chart, capability)
# - spc_plots.py (plot_xbar_r_matplotlib, plot_xbar_plotly)
# - rr_core.py (summary_vs_reference, kappa_matrix_between_operators, confusion_matrix_for_operator)
from clean_io import read_and_normalize
from spc_core import xbar_r, imr, p_chart, c_chart, capability
from spc_plots import plot_xbar_r_matplotlib, plot_xbar_plotly
from rr_core import summary_vs_reference, kappa_matrix_between_operators, confusion_matrix_for_operator

st.set_page_config(page_title="SPC + R&R App", layout="wide")
st.title("SPC + R&R por Atributos — Aplicación robusta (estilo Minitab)")

# Sidebar: opciones globales
st.sidebar.header("Configuración")
mode = st.sidebar.selectbox("Selecciona modo", ["Auto detectar (SPC o R&R)", "Forzar SPC", "Forzar R&R"])
auto_fix = st.sidebar.checkbox("Limpieza automática (auto_fix)", value=True)
show_problems_before_continue = st.sidebar.checkbox("Mostrar problemas antes de continuar (modo seguro)", value=False)
st.sidebar.markdown("---")
st.sidebar.info("Sube archivos Excel o CSV. La app intentará normalizar encabezados y arreglar celdas '16 NG' automáticamente.")

# File uploader
uploaded = st.sidebar.file_uploader("Sube archivo (CSV o Excel)", type=['csv', 'xlsx', 'xls'])

if not uploaded:
    st.info("Sube un archivo para comenzar. Para R&R el archivo debe contener columnas similares a Operador/Pieza/Referencia/Evaluación (la app intentará detectarlas).")
    st.stop()

# Leer & normalizar con clean_io
try:
    df_clean, problems = read_and_normalize(uploaded, auto_fix=auto_fix)
except Exception as e:
    st.error(f"Error al leer/normalizar el archivo: {e}")
    st.stop()

# Mostrar resumen de lo que detectó
st.subheader("Resumen de lectura")
col1, col2 = st.columns([2,1])
with col1:
    st.write("Vista previa del dataset interpretado")
    st.dataframe(df_clean.head(30))
with col2:
    st.write("Metadatos")
    st.write(f"Filas: {df_clean.shape[0]}  |  Columnas: {df_clean.shape[1]}")
    st.write("Problemas detectados:")
    st.write(f"- Filas problemáticas: {len(problems)}")

# Si hay problemas y el usuario quiere revisión manual, mostrar y permitir descarga
if not problems.empty:
    st.warning(f"Se detectaron {len(problems)} filas potencialmente problemáticas.")
    if show_problems_before_continue:
        st.subheader("Filas problemáticas")
        st.dataframe(problems)
    csv_buf = problems.to_csv(index=False).encode('utf-8')
    st.download_button("Descargar filas problemáticas (CSV)", data=csv_buf, file_name='problems.csv', mime='text/csv')

# Si modo seguro y hay problemas, detener hasta que el usuario confirme continuar
if show_problems_before_continue and not problems.empty:
    proceed = st.checkbox("He revisado las filas problemáticas y deseo continuar con el análisis")
    if not proceed:
        st.info("Revise las filas problemáticas o desactive 'Mostrar problemas antes de continuar' en la barra lateral para usar modo automático.")
        st.stop()

# Determinar si R&R o SPC
is_rr = False
is_spc = False

# Si usuario forzó modo:
if mode == "Forzar SPC":
    is_spc = True
elif mode == "Forzar R&R":
    is_rr = True
else:
    # Auto detectar: si el df contiene las 4 columnas clave -> R&R por atributos
    expected_cols = {'Operador', 'Pieza', 'Referencia', 'Evaluación'}
    if expected_cols.issubset(set(df_clean.columns)):
        is_rr = True
    else:
        # Si df_clean tiene >=2 columnas numéricas -> SPC wide
        numeric_cols = df_clean.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) >= 2:
            is_spc = True
        else:
            # fallback: si tiene columnas 0..n-1 treat as wide
            if set(map(str, range(0, df_clean.shape[1]))).intersection(set(df_clean.columns.astype(str))):
                is_spc = True
            else:
                # No se detectó formato claro
                st.error("No se pudo detectar formato SPC ni R&R en el archivo. Revisa problemas.csv o ajusta el archivo.")
                st.stop()

# Modo R&R por atributos
if is_rr:
    st.header("Análisis R&R por Atributos")
    st.markdown("Se calculan: %acuerdo con referencia, Kappa por operador y matriz Kappa entre operadores.")

    # Mostrar resumen por operador vs referencia
    try:
        resumen = summary_vs_reference(df_clean)
        st.subheader("Resumen por operador vs referencia")
        st.dataframe(resumen)
    except Exception as e:
        st.error(f"Error calculando resumen vs referencia: {e}")

    # Matriz Kappa entre operadores
    try:
        st.subheader("Matriz Kappa entre operadores")
        mat = kappa_matrix_between_operators(df_clean)
        st.dataframe(mat)
        if st.button("Mostrar heatmap Kappa"):
            import seaborn as sns
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6,5))
            sns.heatmap(mat.astype(float), annot=True, cmap="coolwarm", vmin=0, vmax=1, ax=ax)
            st.pyplot(fig)
        # Descargar matriz kappa
        csv_mat = mat.to_csv(index=True).encode('utf-8')
        st.download_button("Descargar matriz Kappa (CSV)", data=csv_mat, file_name='kappa_matrix.csv', mime='text/csv')
    except Exception as e:
        st.error(f"Error calculando matriz Kappa: {e}")

    # Matriz de confusión por operador selectable
    try:
        st.subheader("Matriz de confusión por operador (vs referencia)")
        operador_sel = st.selectbox("Selecciona operador para matriz de confusión", df_clean['Operador'].unique())
        cm = confusion_matrix_for_operator(df_clean, operador_sel)
        if not cm.empty:
            st.dataframe(cm)
            csv_cm = cm.to_csv(index=True).encode('utf-8')
            st.download_button(f"Descargar matriz de confusión ({operador_sel})", data=csv_cm, file_name=f'confusion_{operador_sel}.csv', mime='text/csv')
        else:
            st.info("No hay datos suficientes para la matriz de confusión del operador seleccionado.")
    except Exception as e:
        st.error(f"Error generando matriz de confusión: {e}")

# Modo SPC
if is_spc:
    st.header("Análisis SPC (cartas)")

    # Si df_clean parece wide (subgrupos por fila), usarlo directamente
    df_wide = None
    # Si el dataframe contiene columnas numéricas, construimos df_wide con ellas
    num_cols = df_clean.select_dtypes(include=['number']).columns.tolist()
    if len(num_cols) >= 2:
        df_wide = df_clean[num_cols].copy()
    else:
        # If columns are named 0..n-1 string, try convert to numeric
        try:
            df_wide = df_clean.copy()
            df_wide = df_wide.apply(pd.to_numeric, errors='coerce')
            if df_wide.shape[1] < 2:
                st.error("No se detectaron columnas numéricas suficientes para SPC (se requieren al menos 2 mediciones por subgrupo).")
                st.stop()
        except Exception:
            st.error("No se pudo convertir los datos a formato numérico para SPC.")
            st.stop()

    st.subheader("Datos normalizados para SPC (primeras filas)")
    st.dataframe(df_wide.head(20))

    chart = st.selectbox("Tipo de carta", ["Xbar-R", "I-MR", "p-chart", "c-chart", "Capacidad"])
    if chart == "Xbar-R":
        try:
            res = xbar_r(df_wide)
            # matplotlib
            fig = plot_xbar_r_matplotlib(res)
            st.pyplot(fig)
            # plotly interactivo opcional
            if st.checkbox("Mostrar versión interactiva (Plotly)"):
                figp = plot_xbar_plotly(res)
                st.plotly_chart(figp, use_container_width=True)
            # Mostrar resumen
            st.write({
                "Media general": res['media_general'],
                "R̄": res['r_bar'],
                "Xbar UCL": res['xbar_UCL'],
                "Xbar LCL": res['xbar_LCL'],
                "R UCL": res['R_UCL'],
                "R LCL": res['R_LCL'],
                "n típico (mode)": res['n_mode']
            })
            # Descargar PNG
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150)
            buf.seek(0)
            st.download_button("Descargar Xbar-R (PNG)", data=buf, file_name="xbar_r.png", mime="image/png")
        except Exception as e:
            st.error(f"Error generando Xbar-R: {e}")

    elif chart == "I-MR":
        try:
            # Asumimos que la primera columna contiene la serie de individuos si no hay otra opción.
            col_options = list(df_wide.columns)
            col_choice = st.selectbox("Selecciona columna con la serie de individuos", col_options, index=0)
            series = df_wide[col_choice].dropna().astype(float).reset_index(drop=True)
            imr_res = imr(series)
            st.write(imr_res)
            # Plot simple I and MR
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2,1, figsize=(10,7), sharex=True)
            axes[0].plot(series.index, series.values, marker='o')
            axes[0].axhline(imr_res['mu'], color='green')
            axes[0].axhline(imr_res['UCL'], color='red', linestyle='--')
            axes[0].axhline(imr_res['LCL'], color='red', linestyle='--')
            axes[1].plot(imr_res['mr'].index, imr_res['mr'].values, marker='o')
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error generando I-MR: {e}")

    elif chart == "p-chart":
        st.info("Para p-chart necesita columnas: defects (conteo) y n (oportunidades).")
        col_def = st.text_input("Nombre columna defects", value="defects")
        col_n = st.text_input("Nombre columna n (opportunities)", value="n")
        if col_def in df_clean.columns and col_n in df_clean.columns:
            try:
                p_res = p_chart(df_clean[col_def].astype(float).values, df_clean[col_n].astype(float).values)
                st.write(p_res)
            except Exception as e:
                st.error(f"Error calculando p-chart: {e}")
        else:
            st.warning("Columnas definidas no existen en el dataset. Asegúrate de usar los nombres correctos o transformar tu dataset.")

    elif chart == "c-chart":
        st.info("Para c-chart use una columna de conteos por subgrupo (counts).")
        col_count = st.text_input("Nombre columna de counts", value="counts")
        if col_count in df_clean.columns:
            try:
                c_res = c_chart(df_clean[col_count].astype(float).values)
                st.write(c_res)
            except Exception as e:
                st.error(f"Error calculando c-chart: {e}")
        else:
            st.warning("Columna no encontrada.")

    elif chart == "Capacidad":
        try:
            LSL = st.number_input("LSL", value=float(df_wide.min().min()))
            USL = st.number_input("USL", value=float(df_wide.max().max()))
            vals = df_wide.values.flatten().astype(float)
            cap = capability(vals, LSL, USL)
            st.write(cap)
        except Exception as e:
            st.error(f"Error calculando capacidad: {e}")

st.sidebar.markdown("---")
st.sidebar.write("Logs y ayuda")
st.sidebar.write("Si el dataset es muy ruidoso activa 'Mostrar problemas antes de continuar' para revisar manualmente filas conflictivas.")