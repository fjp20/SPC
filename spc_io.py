# clean_io.py
"""
Funciones robustas para leer y normalizar archivos R&R y SPC desordenados.
- read_and_normalize(uploaded_file, auto_fix=True)
  devuelve (df_clean, problems_df)
  df_clean: DataFrame listo para R&R (Operador,Pieza,Referencia,Evaluación)
            o para SPC wide (subgrupos por fila) cuando detecta múltiples columnas numéricas.
  problems_df: DataFrame con filas problemáticas detectadas (puede estar vacío).
"""
import pandas as pd
import numpy as np
import re
from io import BytesIO
from typing import Tuple

def _norm_header(h):
    if pd.isna(h):
        return ""
    return str(h).strip().replace("\n"," ").replace("\r"," ").strip()

def _normalize_label(v):
    if pd.isna(v):
        return np.nan
    s = str(v).strip()
    if s=="":
        return np.nan
    s2 = re.sub(r'\s+',' ', s).upper()
    if s2 in ['OK','O.K.','O K','GOOD','PASS','ACEPTADO','APTO']:
        return 'OK'
    if s2 in ['NG','N.G.','NO GOOD','FAIL','BAD','RECHAZADO','NO APT']:
        return 'NG'
    return s2

def _split_piece_eval(cell):
    """Detecta patrones '16 NG', '2OK', '20-OK' y retorna (pieza:int, eval:str) o (None,None)"""
    if pd.isna(cell):
        return None, None
    s = str(cell).strip()
    # patrón: número + opcional separator + OK/NG (ignorar mayúsc)
    m = re.match(r'^\s*(\d{1,6})\s*[-:;/]?\s*(OK|NG|O\.K\.|N\.G\.|ok|ng)\s*$', s, flags=re.IGNORECASE)
    if m:
        try:
            p = int(m.group(1))
        except:
            p = None
        e = m.group(2).upper().replace('.','')
        if e in ['OK','NG']:
            return p, e
    return None, None

def _col_content_ratio(series, pattern):
    s = series.dropna().astype(str).head(50)
    if s.size == 0:
        return 0.0
    matches = s.str.contains(pattern, regex=True, case=False)
    return matches.sum()/len(s)

def read_raw(uploaded) -> pd.DataFrame:
    """Lee archivo subido (csv/xlsx)."""
    name = getattr(uploaded, "name", "")
    if name.lower().endswith(".csv"):
        return pd.read_csv(uploaded, dtype=str, keep_default_na=False, na_values=[''])
    else:
        return pd.read_excel(uploaded, dtype=str)

def read_and_normalize(uploaded, auto_fix: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Lee y normaliza un archivo. Devuelve (df_clean, problems)
    - Si detecta formato R&R (columns similar a Operador/Pieza/Referencia/Evaluación) produce df_clean con estas columnas.
    - Si detecta formato SPC wide (>=2 columnas numéricas) devuelve DataFrame wide (subgrupos por fila) y problems vacío.
    - problems contiene filas con campos faltantes o ambiguos.
    """
    df0 = read_raw(uploaded)
    # normalizar encabezados
    df0.columns = [_norm_header(c) for c in df0.columns]

    # Primer chequeo: si ya tiene las columnas exactas esperadas
    target = ['Operador','Pieza','Referencia','Evaluación']
    cols_lower = {c.lower(): c for c in df0.columns}
    if all(name.lower() in cols_lower for name in target):
        # mapear columnas a exactas (respetando mayúsc/min)
        mapped = {name: cols_lower[name.lower()] for name in target}
        df = df0.loc[:, mapped.values()].rename(columns={v:k for k,v in mapped.items()})
        # normalizar valores
        df['Referencia'] = df['Referencia'].apply(_normalize_label)
        df['Evaluación'] = df['Evaluación'].apply(_normalize_label)
        df['Pieza'] = pd.to_numeric(df['Pieza'], errors='coerce').astype('Int64')
        # detectar problemas
        problems = df[df[['Operador','Pieza','Referencia','Evaluación']].isna().any(axis=1)].copy()
        return df.reset_index(drop=True), problems.reset_index(drop=True)

    # Si no tiene columnas exactas, intentar heurísticas:
    # 1) Si hay >=2 columnas numéricas -> SPC wide
    numeric_cols = []
    for c in df0.columns:
        # consideramos columna numérica si la mayoría de sus primeras 30 no-nulos parsean a float
        col = df0[c].dropna().astype(str).head(60)
        if col.size==0:
            continue
        parsable = sum(1 for v in col if re.match(r'^\s*-?\d+(\.\d+)?\s*$', v))
        if parsable/len(col) > 0.7:
            numeric_cols.append(c)
    if len(numeric_cols) >= 2:
        # devolver DataFrame wide con columnas renombradas a 0..n-1 (pandas numeric)
        wide = df0[numeric_cols].copy()
        wide.columns = list(range(wide.shape[1]))
        wide = wide.applymap(lambda x: np.nan if (isinstance(x,str) and x.strip()=='') else x)
        wide = wide.apply(pd.to_numeric, errors='coerce')
        return wide.reset_index(drop=True), pd.DataFrame(columns=['issue'])  # no problems in R&R sense

    # 2) Intentar mapear columnas por contenido: buscar columna operador (texto), pieza (numérica), referencia/evaluación (OK/NG)
    candidate = { 'op': None, 'piece': None, 'ref': None, 'eval': None }
    for c in df0.columns:
        s = df0[c].astype(str).replace('','NaN').dropna().head(60)
        if s.size == 0:
            continue
        # ratio OK/NG
        okng_ratio = _col_content_ratio(df0[c], r'\b(OK|NG)\b')
        num_ratio = _col_content_ratio(df0[c], r'^\s*\d+\s*$')
        text_ratio = _col_content_ratio(df0[c], r'[A-Za-z]')
        # heurísticas
        if okng_ratio > 0.6 and candidate['eval'] is None:
            candidate['eval'] = c
            continue
        if num_ratio > 0.7 and candidate['piece'] is None:
            candidate['piece'] = c
            continue
        if text_ratio > 0.6 and candidate['op'] is None:
            candidate['op'] = c
            continue
        if okng_ratio > 0.3 and candidate['ref'] is None:
            candidate['ref'] = c

    # Mapear encontrados; si faltan columnas, reconstruir fila por fila intentando separar patrones '16 NG'
    df_rows = []
    for idx, row in df0.iterrows():
        op = row[candidate['op']] if candidate['op'] else None
        piece = row[candidate['piece']] if candidate['piece'] else None
        ref = row[candidate['ref']] if candidate['ref'] else None
        eva = row[candidate['eval']] if candidate['eval'] else None

        # si alguno es vacío intentar extraer de otras celdas
        # buscar celdas con pattern "num + OK/NG"
        if pd.isna(piece) or pd.isna(eva):
            for c in df0.columns:
                p,e = _split_piece_eval(row[c])
                if p is not None and e is not None:
                    if pd.isna(piece):
                        piece = p
                    if pd.isna(eva):
                        eva = e
        # si referencia falta pero hay una celda OK/NG en la fila
        if pd.isna(ref):
            for c in df0.columns:
                v = row[c]
                if pd.isna(v): continue
                if re.fullmatch(r'(?i)\s*(OK|NG)\s*', str(v).strip()):
                    ref = str(v).strip()
                    break
        # si operador falta, buscar la primera cadena que no sea OK/NG ni número
        if pd.isna(op):
            for c in df0.columns:
                v = row[c]
                if pd.isna(v): continue
                s = str(v).strip()
                if re.search(r'[A-Za-z]', s) and not re.match(r'^\s*\d+\s*(OK|NG)?\s*$', s, flags=re.IGNORECASE):
                    op = s
                    break

        df_rows.append({'Operador': op, 'Pieza': piece, 'Referencia': ref, 'Evaluación': eva})

    df_clean = pd.DataFrame(df_rows, columns=['Operador','Pieza','Referencia','Evaluación'])
    # Normalizar labels
    df_clean['Referencia'] = df_clean['Referencia'].apply(_normalize_label)
    df_clean['Evaluación'] = df_clean['Evaluación'].apply(_normalize_label)
    # Pieza a entero cuando sea posible
    df_clean['Pieza'] = pd.to_numeric(df_clean['Pieza'], errors='coerce').astype('Int64')

    # Detectar problemas: filas con Operador vacío o con ambas Referencia/Evaluación vacías
    mask_problem = df_clean['Operador'].isna() | (df_clean['Referencia'].isna() & df_clean['Evaluación'].isna())
    problems = df_clean[mask_problem].copy()
    problems['issue'] = problems.apply(lambda r: 'missing' if pd.isna(r['Operador']) else 'no_ref_eval', axis=1)

    # Si auto_fix es True, intentar inferir referencia = evaluación cuando referencia falta but evaluation present
    if auto_fix:
        missing_ref = df_clean['Referencia'].isna() & df_clean['Evaluación'].notna()
        df_clean.loc[missing_ref, 'Referencia'] = df_clean.loc[missing_ref, 'Evaluación']

    # Devolver df_clean sin filas totalmente vacías
    df_final = df_clean.dropna(how='all').reset_index(drop=True)
    problems = problems.reset_index(drop=True)
    return df_final, problems