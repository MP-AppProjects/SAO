"""
sao_core.py -- warstwa analityczna i eksporty dla SAO.

Zawiera czyste funkcje (bez UI Streamlita poza st.session_state/cache):
wczytywanie danych, transformacje (rekodowania, czyszczenie, segmentacje),
budowanie tabel, statystyke (regresja, ANOVA, korelacje, czynnikowa,
conjoint, maxdiff, wagi RIM) oraz eksport do Excela.

Wydzielone z generator.py. Importowane przez `import sao_core; from sao_core import *`.
Plik MUSI pozostac czystym ASCII (polskie znaki jako sekwencje ucieczki).

original_cols: wstrzykiwana przez generator.py po wczytaniu danych
(uzywana w get_var_display_name do oznaczania zmiennych pochodnych).
"""
import streamlit as st
import pandas as pd
import numpy as np
import pyreadstat
import tempfile
import io
import json
import re
import copy
import string
import datetime
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
import sqlite3
import hashlib
import secrets
import uuid
import os
import ipaddress
import urllib.request
import urllib.error
import time


original_cols = None


@st.cache_data
def load_spss_data(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".sav") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    df, meta = pyreadstat.read_sav(tmp_path)
    df_labeled = df.copy()

    def _code_to_str(v):
        # niezmapowany kod liczbowy -> czytelny tekst (bez ".0" dla calkowitych)
        try:
            f = float(v)
            return str(int(f)) if f.is_integer() else str(f)
        except (TypeError, ValueError):
            return str(v)

    for col, labels in meta.variable_value_labels.items():
        if col in df_labeled.columns:
            # Etykiety SPSS sa tekstem. Kategorie kolumny 'category' MUSZA byc
            # jednorodne (tekst), inaczej pyarrow rzuca ArrowTypeError przy
            # serializacji (mieszanka str etykiet + float niezmapowanych kodow).
            ordered_cats = [str(labels[k]) for k in sorted(labels.keys())]
            mapped = df_labeled[col].map(labels).astype(object)
            # niezmapowane kody (bez etykiety) -> tekstowa reprezentacja kodu
            _unl = mapped.isna() & df_labeled[col].notna()
            mapped[_unl] = df_labeled[col][_unl].map(_code_to_str)
            existing = mapped.dropna().unique()
            extra = [e for e in existing if e not in ordered_cats]
            df_labeled[col] = pd.Categorical(mapped, categories=ordered_cats + extra, ordered=True)
    return df, df_labeled, meta


# ------------------------------------------------------------------
# Excel compatibility layer -- mimics pyreadstat meta interface
# ------------------------------------------------------------------
class ExcelMeta:
    """Thin compatibility wrapper so all meta_orig references work for Excel data."""
    def __init__(self, columns, col_labels=None):
        col_labels = col_labels or {}
        self.column_names_to_labels = {c: col_labels.get(c, c) for c in columns}
        self.variable_value_labels  = {}   # no SPSS value labels for Excel

    def get(self, key, default=None):
        return self.column_names_to_labels.get(key, default)


def _apply_tabular_types(df, col_type_overrides, custom_missing):
    """Wspolna logika typowania dla danych tabelarycznych (Excel / CSV).

    df:                 surowy DataFrame (naglowek w wierszu 0, juz wczytany)
    col_type_overrides: {col: 'numeric'|'categorical'} -- recznie wymuszony typ
    custom_missing:     {col: [val, ...]} -- wartosci traktowane jako braki
    Zwraca (df_raw, df_labeled, ExcelMeta). Tekstowe kolumny wymuszone na
    'numeric' sa kodowane kolejnymi liczbami 1,2,3... (mapa -> etykiety wartosci).
    """
    # Clean column names
    df.columns = [str(c).strip() for c in df.columns]

    # Auto-detect and apply types
    df_raw = df.copy()
    df_labeled = df.copy()

    # Collect text\u2192numeric mappings to store as value labels
    _text_to_num_maps = {}  # {col: {1: 'Kobieta', 2: 'M\u0119\u017cczyzna', ...}}

    def _missing_str_set(col):
        """Build set of string representations of missing values for a column."""
        m_vals = custom_missing.get(col, [])
        s = set()
        for v in m_vals:
            s.add(str(v))
            try:
                s.add(str(int(float(v))))
                s.add(str(float(v)))
            except (ValueError, TypeError):
                pass
        return s

    for col in df.columns:
        override = col_type_overrides.get(col)
        if override == 'numeric':
            # Try direct numeric conversion first
            numeric_attempt = pd.to_numeric(df_raw[col], errors='coerce')
            non_null = df_raw[col].dropna()
            survived = numeric_attempt.notna().sum()
            if len(non_null) == 0 or survived / len(non_null) >= 0.9:
                df_raw[col]    = numeric_attempt
                df_labeled[col] = numeric_attempt.copy()
            else:
                # Column is text \u2014 encode as consecutive integers 1, 2, 3...
                # Skip values that are defined as missing
                miss_strs = _missing_str_set(col)
                unique_vals = sorted(
                    [v for v in df_raw[col].dropna().unique()
                     if str(v) not in miss_strs],
                    key=lambda x: str(x)
                )
                code_map  = {v: i + 1 for i, v in enumerate(unique_vals)}
                label_map = {i + 1: str(v) for i, v in enumerate(unique_vals)}
                df_raw[col]    = df_raw[col].map(code_map)   # missing vals \u2192 NaN
                df_labeled[col] = df_raw[col].copy()
                _text_to_num_maps[col] = label_map
        elif override == 'categorical':
            df_raw[col]    = df_raw[col].astype(str).where(df_raw[col].notna(), np.nan)
            df_labeled[col] = df_raw[col].copy()
        else:
            # Auto: try numeric first
            numeric_attempt = pd.to_numeric(df_raw[col], errors='coerce')
            non_null = df_raw[col].dropna()
            if len(non_null) > 0:
                numeric_rate = numeric_attempt.notna().sum() / len(non_null)
                if numeric_rate >= 0.9:
                    df_raw[col]    = numeric_attempt
                    df_labeled[col] = numeric_attempt.copy()
                # else: leave as object (categorical)

    meta = ExcelMeta(df.columns.tolist())
    meta._text_to_num_maps = _text_to_num_maps
    return df_raw, df_labeled, meta


@st.cache_data
def load_excel_data(file_bytes, sheet_name, col_type_overrides_json="{}", custom_missing_json="{}"):
    """
    Load an Excel file (from raw bytes -- hashable, poprawne kluczowanie cache).
    Row 1 = variable names, row 2+ = data.
    col_type_overrides_json: JSON string of {col: 'numeric'|'categorical'}
    custom_missing_json:     JSON string of {col: [val, ...]} missing values per column
    Returns (df_raw, df_labeled, ExcelMeta)
    """
    col_type_overrides = json.loads(col_type_overrides_json)
    custom_missing     = json.loads(custom_missing_json)
    df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name, header=0)
    return _apply_tabular_types(df, col_type_overrides, custom_missing)


@st.cache_data
def load_csv_data(file_bytes, sep=";", decimal=",", encoding="utf-8", header_row=0,
                  col_type_overrides_json="{}", custom_missing_json="{}"):
    """
    Load a CSV file (from raw bytes). Separator, separator dziesietny i kodowanie
    podawane jawnie (wykrywane w kreatorze importu). Detekcja typow jak w Excelu.
    Returns (df_raw, df_labeled, ExcelMeta)
    """
    col_type_overrides = json.loads(col_type_overrides_json)
    custom_missing     = json.loads(custom_missing_json)
    df = pd.read_csv(io.BytesIO(file_bytes), sep=sep, decimal=decimal,
                     encoding=encoding, header=header_row,
                     skipinitialspace=True, engine="python")
    return _apply_tabular_types(df, col_type_overrides, custom_missing)


def sniff_csv_dialect(file_bytes, n_bytes=8192):
    """Heurystycznie wykryj kodowanie, separator i separator dziesietny pliku CSV.
    Zwraca dict {encoding, sep, decimal}. Bezpieczne wartosci domyslne przy bledzie.
    """
    # 1. kodowanie -- sprobuj utf-8, potem typowe dla polskiego Windows
    sample = None
    enc_used = "utf-8"
    for enc in ("utf-8-sig", "utf-8", "cp1250", "iso-8859-2", "latin1"):
        try:
            sample = file_bytes[:n_bytes].decode(enc)
            enc_used = "utf-8" if enc == "utf-8-sig" else enc
            break
        except (UnicodeDecodeError, LookupError):
            continue
    if sample is None:
        return {"encoding": "utf-8", "sep": ";", "decimal": ","}
    # 2. separator kolumn -- najczestszy z kandydatow w pierwszej linii
    first_line = sample.splitlines()[0] if sample.splitlines() else ""
    counts = {c: first_line.count(c) for c in [";", ",", "\t", "|"]}
    sep = max(counts, key=counts.get) if max(counts.values()) > 0 else ";"
    # 3. separator dziesietny -- jesli sep=';' i w probce sa liczby z przecinkiem
    import re as _re
    decimal = ","
    if sep == ",":
        decimal = "."          # przecinek to separator kolumn -> kropka dziesietna
    elif _re.search(r"\d,\d", sample):
        decimal = ","
    else:
        decimal = "."
    return {"encoding": enc_used, "sep": sep, "decimal": decimal}


def auto_detect_mrs(df_raw):
    """Wykryj zestawy wielokrotnego wyboru (MRS).

    Dwie sciezki detekcji:
    1. Krotkie nazwy z separatorem (Q1_1, Q1_2... lub Q1.1, Q1.2...):
       grupowanie przez rsplit na '_' lub '.'
    2. Dlugie nazwy (Excel: pelny tekst pytania) ze wspolnym prefiksem
       o dlugosci >= 8 znakow -- obsluguje kolumny typu
       "Satysfakcja - Szybkosc", "Satysfakcja - Jakosc" itp.
    """
    from itertools import combinations as _comb
    binary_cols = [c for c in df_raw.columns
                   if set(df_raw[c].dropna().unique()).issubset({0, 1})
                   and len(set(df_raw[c].dropna().unique())) > 0]
    mrs_candidates = defaultdict(list)
    for col in binary_cols:
        s = str(col)
        if '_' in s:
            prefix = s.rsplit('_', 1)[0]
        elif '.' in s and not s.endswith('.'):
            prefix = s.rsplit('.', 1)[0]
        else:
            prefix = None
        if prefix:
            mrs_candidates[prefix].append(col)
    result = {k: v for k, v in mrs_candidates.items() if len(v) > 1}

    # Sciezka 2: wspolny prefiks dla niezgrupowanych kolumn (Excel)
    _SEP = set(' -\u2014/|:.')
    already = set(c for v in result.values() for c in v)
    remaining = [c for c in binary_cols if c not in already]
    if len(remaining) >= 2:
        def _lcp(a, b):
            i = 0
            while i < min(len(a), len(b)) and a[i] == b[i]:
                i += 1
            return a[:i].rstrip(''.join(_SEP) + ' ')

        lcp_groups = defaultdict(set)
        for c1, c2 in _comb(remaining, 2):
            p = _lcp(str(c1), str(c2))
            if len(p) >= 8:
                lcp_groups[p].add(c1)
                lcp_groups[p].add(c2)
        for prefix, cols in lcp_groups.items():
            if len(cols) >= 2:
                key = prefix[:60]
                if key not in result:
                    result[key] = sorted(cols)
    return result

def auto_detect_matrix(df_raw):
    """
    Detect matrix/battery questions using two signals combined:
      1. Shared variable-name prefix (Q1_1, Q1_2... or Q1a, Q1b...)
      2. Compatible value set \u2014 subquestions share a scale, but tolerate
         missing categories (e.g. if Q1_3 respondents never picked '5'
         the column still belongs to the same matrix).

    Returns dict {battery_name: [sorted columns...]} with >=2 subquestions.
    """
    import re
    binary_cols = set(c for c in df_raw.columns
                      if set(df_raw[c].dropna().unique()).issubset({0, 1}))
    candidates = [c for c in df_raw.columns if c not in binary_cols]

    def _extract_prefix(col):
        s = str(col).strip()
        m = re.match(r"^(.+?)[_.\-](\w+)$", s)
        if m:
            prefix, suffix = m.group(1), m.group(2)
            if suffix.isdigit() or (len(suffix) <= 3 and re.match(r"^[a-zA-Z0-9]+$", suffix)):
                return prefix
        m = re.match(r"^(.+?)(\d+)$", s)
        if m and len(m.group(1)) >= 1:
            return m.group(1)
        m = re.match(r"^(.+?)([a-zA-Z])$", s)
        if m and len(m.group(1)) >= 2:
            return m.group(1)
        return None

    def _value_set(col_series):
        return frozenset(col_series.dropna().unique())

    def _compatible(cluster_union, candidate_set):
        """Candidate belongs to cluster if its values are subset of cluster union,
        OR overlap is >=60% of the smaller set (tolerates missing categories)."""
        if not cluster_union or not candidate_set:
            return False
        # Perfect subset (candidate is a subset of cluster scale, with missing cats)
        if candidate_set.issubset(cluster_union):
            return True
        # Cluster is itself a subset of candidate - candidate has all cluster vals + extras
        if cluster_union.issubset(candidate_set):
            return True
        # Partial overlap test
        inter = cluster_union & candidate_set
        smaller = min(len(cluster_union), len(candidate_set))
        return (len(inter) / smaller) >= 0.6

    prefix_groups = defaultdict(list)
    for col in candidates:
        p = _extract_prefix(col)
        if p:
            prefix_groups[p].append(col)

    result = {}
    for prefix, cols in prefix_groups.items():
        if len(cols) < 2:
            continue
        col_sets = {}
        for c in cols:
            s = _value_set(df_raw[c])
            if 2 <= len(s) <= 15:
                col_sets[c] = s
        if len(col_sets) < 2:
            continue

        # Greedy clustering: seed with largest value set (most complete scale),
        # absorb compatible columns, update union, repeat
        remaining = dict(col_sets)
        clusters = []
        while remaining:
            seed = max(remaining.keys(), key=lambda c: len(remaining[c]))
            seed_set = remaining.pop(seed)
            cluster = [seed]
            cluster_union = set(seed_set)
            changed = True
            while changed:
                changed = False
                for c in list(remaining.keys()):
                    if _compatible(frozenset(cluster_union), remaining[c]):
                        cluster.append(c)
                        cluster_union |= remaining[c]
                        remaining.pop(c)
                        changed = True
            if len(cluster) >= 2:
                clusters.append(cluster)

        if not clusters:
            continue
        if len(clusters) == 1:
            result[prefix] = sorted(clusters[0])
        else:
            for i, cl in enumerate(clusters, 1):
                result[f"{prefix}__{i}"] = sorted(cl)

    # Sciezka 2: wspolny prefiks dla kolumn z dlugimi nazwami (Excel).
    # Dla kolumn ktore _extract_prefix() nie potrafilo sklasyfikowac (None)
    # szukamy par z co najmniej 8-znakowym wspolnym prefiksem i zgodnym
    # zestawem wartosci.
    from itertools import combinations as _comb2
    _SEP2 = set(' -\u2014/|:.')
    already_grouped = set(c for v in result.values() for c in v)
    remaining_long = [c for c in candidates if c not in already_grouped]
    if len(remaining_long) >= 2:
        def _lcp2(a, b):
            i = 0
            while i < min(len(a), len(b)) and a[i] == b[i]:
                i += 1
            return a[:i].rstrip(''.join(_SEP2) + ' ')

        lcp_groups2 = defaultdict(set)
        for c1, c2 in _comb2(remaining_long, 2):
            p = _lcp2(str(c1), str(c2))
            if len(p) >= 8:
                lcp_groups2[p].add(c1)
                lcp_groups2[p].add(c2)
        for prefix2, group_cols in lcp_groups2.items():
            if len(group_cols) < 2:
                continue
            col_sets2 = {}
            for c in sorted(group_cols):
                s = _value_set(df_raw[c])
                if 2 <= len(s) <= 15:
                    col_sets2[c] = s
            if len(col_sets2) < 2:
                continue
            key2 = prefix2[:50]
            if key2 not in result:
                result[key2] = sorted(col_sets2.keys())
    return result

def _apply_value_order(values_iter, var_name):
    """Reorder values_iter according to user-defined st.session_state.value_orders[var_name].
    Values not present in the user-defined order are appended at the end
    in their original (input) order. Returns a list with original element types preserved."""
    try:
        order = st.session_state.value_orders.get(var_name) or []
    except Exception:
        order = []
    values_list = list(values_iter)
    if not order:
        return values_list
    str_to_orig = {}
    for v in values_list:
        sv = str(v)
        if sv not in str_to_orig:
            str_to_orig[sv] = v
    seen = set()
    result = []
    for o in order:
        os_ = str(o)
        if os_ in str_to_orig and os_ not in seen:
            result.append(str_to_orig[os_])
            seen.add(os_)
    for v in values_list:
        sv = str(v)
        if sv not in seen:
            result.append(v)
            seen.add(sv)
    return result


def build_matrix_table(df, df_raw, matrix_cols, var_labels, weights, meta_vvl, custom_val_labels=None):
    """
    Build a matrix frequency table with TRANSPOSED layout:
      Rows    = scale values / categories  (e.g. 1, 2, 3 ... or 'Tak','Nie')
      Columns = subquestions (variable names / labels)
      Cells   = N and % per subquestion \u00d7 value combination

    Also appends a combined 'Baza (N) / Suma (%)' summary row with N and % side by side.
    custom_val_labels: {var_name: {str(code): new_label}} overrides display labels.

    Returns: df_out, all_cats (display labels), sub_labels
    """
    w = weights
    if custom_val_labels is None:
        custom_val_labels = {}

    # -- 1. Collect all unique raw category values across the battery --
    raw_cats_set = []
    for col in matrix_cols:
        series = (df[col] if col in df.columns else df_raw[col]).dropna()
        for cat in series.unique():
            cat_str = str(cat)
            if cat_str not in raw_cats_set:
                raw_cats_set.append(cat_str)

    # -- 1b. Sort categories in SPSS-defined order (Categorical > vvl > numeric > alpha) --
    _ordered, _seen_o = [], set()
    # First choice: Categorical order preserved from SPSS load (df_labeled uses ordered=True)
    for col in matrix_cols:
        if col in df.columns and hasattr(df[col], 'cat'):
            for _cv in df[col].cat.categories:
                _cs = str(_cv)
                if _cs in raw_cats_set and _cs not in _seen_o:
                    _ordered.append(_cs)
                    _seen_o.add(_cs)
            if _ordered:
                break
    # Second choice: SPSS variable_value_labels insertion order
    if not _ordered:
        for col in matrix_cols:
            _vvl = meta_vvl.get(col, {})
            if _vvl:
                for _code, _lbl in _vvl.items():
                    for _cs in (str(_lbl), str(_code)):
                        if _cs in raw_cats_set and _cs not in _seen_o:
                            _ordered.append(_cs)
                            _seen_o.add(_cs)
                break
    # Append anything not yet covered (unlabelled codes etc.)
    for _cs in raw_cats_set:
        if _cs not in _seen_o:
            _ordered.append(_cs)
    # Fallback: numeric or alphabetical
    if _ordered:
        raw_cats_sorted = _ordered
    else:
        try:
            raw_cats_sorted = sorted(raw_cats_set, key=lambda x: float(x))
        except (ValueError, TypeError):
            raw_cats_sorted = sorted(raw_cats_set)
    # User-defined value order overrides everything (uses first matrix col with an order)
    try:
        _vo_dict = st.session_state.value_orders
    except Exception:
        _vo_dict = {}
    for _mc in matrix_cols:
        if _mc in _vo_dict and _vo_dict[_mc]:
            raw_cats_sorted = _apply_value_order(raw_cats_sorted, _mc)
            break

    # -- 2. Build display label map for categories (apply custom_val_labels) --
    # Since value labels may differ per column, build a unified best-effort map
    # using the first column's custom/SPSS labels as reference, then merge.
    cat_display = {}   # raw_str ? display_str
    for raw_str in raw_cats_sorted:
        cat_display[raw_str] = raw_str   # default: show raw value

    # Apply SPSS value labels from the first column that has them
    for col in matrix_cols:
        spss_vvl = meta_vvl.get(col, {})
        col_custom = custom_val_labels.get(col, {})
        for raw_str in raw_cats_sorted:
            # Custom label overrides SPSS label
            if raw_str in col_custom:
                cat_display[raw_str] = col_custom[raw_str]
            elif raw_str not in col_custom:
                # Try SPSS numeric key
                try:
                    num_key = float(raw_str)
                    if num_key in spss_vvl and cat_display[raw_str] == raw_str:
                        cat_display[raw_str] = spss_vvl[num_key]
                except (ValueError, TypeError):
                    if raw_str in spss_vvl and cat_display[raw_str] == raw_str:
                        cat_display[raw_str] = spss_vvl[raw_str]

    display_cats = [cat_display[r] for r in raw_cats_sorted]   # display labels, in order

    # -- 3. For each subquestion: count N and % per category --
    sub_labels = []
    data_n   = {}
    data_pct = {}
    data_base = {}

    for col in matrix_cols:
        sub_lbl = var_labels.get(col, col)
        if sub_lbl in sub_labels:
            sub_lbl = f"{sub_lbl} [{col}]"
        sub_labels.append(sub_lbl)

        series = (df[col] if col in df.columns else df_raw[col])
        missing_mask = series.isna()
        base_w = float(w[~missing_mask].sum())
        data_base[sub_lbl] = base_w

        counts = {}
        for raw_str, disp_str in zip(raw_cats_sorted, display_cats):
            mask = (series.astype(str) == raw_str) & (~missing_mask)
            counts[disp_str] = float(w[mask].sum())
        data_n[sub_lbl]   = counts
        data_pct[sub_lbl] = {disp: (v / base_w * 100 if base_w > 0 else 0.0)
                             for disp, v in counts.items()}

    # -- 4. Build output DataFrame --
    # Columns interleaved: SubA [N], SubA [%], SubB [N], SubB [%], ...
    interleaved_cols = []
    for lbl in sub_labels:
        interleaved_cols.append(f"{lbl} [N]")
        interleaved_cols.append(f"{lbl} [%]")

    # Rows: one per display category + one combined summary row "Baza (N) / Suma (%)"
    SUMMARY_ROW = "Baza (N) / Suma (%)"
    all_rows = display_cats + [SUMMARY_ROW]
    df_out = pd.DataFrame(index=all_rows, columns=interleaved_cols, dtype=object)

    for sub_lbl in sub_labels:
        col_sum_pct = 0.0
        for disp_str in display_cats:
            n_val   = data_n[sub_lbl][disp_str]
            pct_val = data_pct[sub_lbl][disp_str]
            df_out.loc[disp_str, f"{sub_lbl} [N]"] = n_val
            df_out.loc[disp_str, f"{sub_lbl} [%]"] = pct_val
            col_sum_pct += pct_val
        # Summary row: N = base respondents, % = sum of percentages (100%)
        df_out.loc[SUMMARY_ROW, f"{sub_lbl} [N]"] = data_base[sub_lbl]
        df_out.loc[SUMMARY_ROW, f"{sub_lbl} [%]"] = round(col_sum_pct, 1)

    return df_out, display_cats, sub_labels

def apply_segmentations(df_raw, df, meta_labels, segmentations_list):
    for seg in segmentations_list:
        cols, k, name = seg['vars'], seg['k'], seg['name']
        X = df_raw[cols].copy()
        for c in cols:
            m_val = X[c].mean()
            X[c] = X[c].fillna(m_val if pd.notna(m_val) else 0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled) + 1
        df_raw[name] = clusters
        df[name] = [f"Segment {c}" for c in clusters]
        df[name] = pd.Categorical(df[name], categories=[f"Segment {i}" for i in range(1, k + 1)], ordered=True)
        meta_labels[name] = f"Segmentacja K-Means ({k} grup)"

def apply_recodings(df_raw, df, var_labels, recodings_list):
    """Apply variable recodings stored in session state."""
    import streamlit as _st
    _cvar = _st.session_state.get('custom_var_labels', {})
    for rec in recodings_list:
        src = rec['source']
        new_name = rec['new_name']
        mapping = rec['mapping']
        label = rec.get('label', new_name)
        output_type = rec.get('output_type', 'auto')
        if src not in df_raw.columns:
            continue
        src_series = df_raw[src].copy().astype(str).str.strip()
        lookup = {str(k).strip(): str(v) for k, v in mapping.items()}
        new_col = src_series.map(lookup)
        if output_type == 'numeric':
            df_raw[new_name] = pd.to_numeric(new_col, errors='coerce')
            df[new_name] = df_raw[new_name].copy()
        elif output_type == 'text':
            df_raw[new_name] = new_col
            df[new_name] = new_col
        else:
            numeric_attempt = pd.to_numeric(new_col, errors='coerce')
            if new_col.dropna().empty or numeric_attempt.notna().sum() == new_col.notna().sum():
                df_raw[new_name] = numeric_attempt
                df[new_name] = numeric_attempt.copy()
            else:
                df_raw[new_name] = new_col
                df[new_name] = new_col
        var_labels[new_name] = _cvar.get(new_name, label)

def apply_hclust_columns(df_raw, df, var_labels, hclust_results):
    """Re-apply stored hierarchical cluster columns to df_raw/df on every rerun.
    Without this, cluster columns created by hclust disappear after a rerun
    because df_raw is rebuilt from df_orig_raw each time.
    Uses vectorized assignment for performance."""
    for rec in hclust_results:
        vname = rec.get('var_name')
        labels_data = rec.get('labels_data', {})
        if not vname or not labels_data:
            continue
        # Build a Series from labels_data with proper index types
        # Try converting all keys to int first (row indexes usually are int)
        try:
            idx_converted = {int(k): v for k, v in labels_data.items()}
        except (ValueError, TypeError):
            idx_converted = labels_data
        labels_series = pd.Series(idx_converted, dtype='float64')
        # Filter to indices that exist in df_raw
        labels_series = labels_series.reindex(df_raw.index.intersection(labels_series.index))
        if labels_series.empty:
            continue
        # Vectorized assignment \u2014 one operation per dataframe
        df_raw[vname] = labels_series.reindex(df_raw.index)
        df[vname]     = labels_series.reindex(df.index).apply(
            lambda x: f"Skupienie {int(x)}" if pd.notna(x) else np.nan
        )
        n_clusters = rec.get('n_clusters', '?')
        method = rec.get('method', '?')
        var_labels[vname] = f"Skupienia hierarchiczne ({n_clusters} grup, {method})"

def apply_cleaning_ops(df_raw, df, cleaning_ops_list):
    """
    Apply stored cleaning operations in-place to the original columns.
    Each entry in cleaning_ops_list:
      {
        'cols': [col1, col2, ...],   # columns to clean
        'ops':  {                    # which operations to apply
            'strip': bool,
            'dbl_sp': bool,
            'tabs': bool,
            'newlines': bool,
            'quotes': bool,
            'case': 'none'|'upper'|'lower'|'title',
            'special': bool,
        }
      }
    """
    QUOTES_MAP = [
        ('\u201c', '"'), ('\u201d', '"'), ('\u201e', '"'),
        ('\u2018', "'"), ('\u2019', "'"), ('\u201a', "'"),
    ]
    for entry in cleaning_ops_list:
        ops  = entry.get('ops', {})
        cols = entry.get('cols', [])
        for col in cols:
            if col not in df_raw.columns:
                continue
            was_null = df_raw[col].isna()
            series = df_raw[col].astype(str).copy()

            if ops.get('strip'):
                series = series.str.strip()
            if ops.get('dbl_sp'):
                series = series.str.replace(r' {2,}', ' ', regex=True)
            if ops.get('tabs'):
                series = series.str.replace('\t', ' ', regex=False)
            if ops.get('newlines'):
                series = series.str.replace(r'[\n\r]', ' ', regex=True)
            if ops.get('quotes'):
                for old_q, new_q in QUOTES_MAP:
                    series = series.str.replace(old_q, new_q, regex=False)
            case = ops.get('case', 'none')
            if   case == 'upper': series = series.str.upper()
            elif case == 'lower': series = series.str.lower()
            elif case == 'title': series = series.str.title()
            if ops.get('special'):
                series = series.str.replace(r'[^\w\s]', '', regex=True)
                series = series.str.replace('_', '', regex=False)

            # Write back, preserving original NaN positions.
            # Convert Categorical columns to object first \u2014 cleaned values
            # may not exist in the original category set.
            if hasattr(df_raw[col], 'cat'):
                df_raw[col] = df_raw[col].astype(object)
            if hasattr(df[col], 'cat'):
                df[col] = df[col].astype(object)

            df_raw.loc[~was_null, col] = series[~was_null]
            df.loc[~was_null, col]     = series[~was_null]


def get_var_display_name(var_name, meta):
    # MRS / matrix set virtual names
    if 'mrs_sets' in st.session_state and var_name in st.session_state.mrs_sets:
        return f"[{var_name}] Zestaw Wielokrotnych Odpowiedzi"
    if 'matrix_sets' in st.session_state and var_name in st.session_state.matrix_sets:
        return f"[{var_name}] Pytanie matrycowe"

    # Resolve label
    label = meta.get(var_name, var_name) if isinstance(meta, dict) else meta.column_names_to_labels.get(var_name, var_name)

    # Mark derived (added during session) with a visible prefix
    # 'original_cols' is set at load time; fall back gracefully if not yet defined
    try:
        is_derived = (original_cols is not None) and (var_name not in original_cols)
    except Exception:
        is_derived = False

    prefix = "[+] " if is_derived else ""
    return f"{prefix}[{var_name}] {label}"

def get_weighted_stats(x, w):
    mask = ~np.isnan(x)
    x_valid, w_valid = x[mask], w[mask]
    sum_w = w_valid.sum()
    if sum_w == 0:
        return np.nan, np.nan, 0
    mean = (x_valid * w_valid).sum() / sum_w
    var = (w_valid * (x_valid - mean) ** 2).sum() / sum_w
    ess = (sum_w ** 2) / (w_valid ** 2).sum() if (w_valid ** 2).sum() > 0 else 0
    return mean, var, ess

def apply_means_sig_testing(df_means, df_vars, df_ess):
    """Parami T-test Welcha (95%) miedzy kolumnami df_means.

    Zwraca (sig_df, col_letters):
      sig_df     - DataFrame stringow z literami istotnych roznic
      col_letters - slownik {col: litera}
    Kolumna "Og\u00f3\u0142em" jest wykluczona z por\u00f3wnan (nie jest niezalezna).
    """
    SKIP_COLS = {"Og\u00f3\u0142em"}
    cols = [c for c in df_means.columns if c not in SKIP_COLS]
    all_cols = list(df_means.columns)
    letters = list(string.ascii_uppercase)
    col_letters = {c: letters[i % 26] * (i // 26 + 1) for i, c in enumerate(cols)}
    sig_df = pd.DataFrame("", index=df_means.index, columns=all_cols)

    if len(cols) < 2:
        return sig_df, col_letters

    def _flt(val):
        """Bezpiecznie zamien na float; NaN jesli niepowodzenie."""
        try:
            fv = float(val)
            return fv if np.isfinite(fv) else np.nan
        except (TypeError, ValueError):
            return np.nan

    for r in df_means.index:
        for i, c1 in enumerate(cols):
            for j, c2 in enumerate(cols):
                if i >= j:
                    continue
                try:
                    m1 = _flt(df_means.loc[r, c1])
                    m2 = _flt(df_means.loc[r, c2])
                    v1 = _flt(df_vars.loc[r, c1])
                    v2 = _flt(df_vars.loc[r, c2])
                    n1 = _flt(df_ess.loc[r, c1])
                    n2 = _flt(df_ess.loc[r, c2])
                    if (np.isnan(m1) or np.isnan(m2)
                            or np.isnan(v1) or np.isnan(v2)
                            or np.isnan(n1) or np.isnan(n2)):
                        continue
                    if n1 < 2 or n2 < 2:
                        continue
                    # Welch: se = sqrt(v1/n1 + v2/n2)
                    se2 = v1 / n1 + v2 / n2
                    if se2 <= 0:
                        continue
                    se = np.sqrt(se2)
                    if se == 0:
                        continue
                    t_stat = (m1 - m2) / se
                    if np.isnan(t_stat) or not np.isfinite(t_stat):
                        continue
                    if t_stat > 1.96:
                        sig_df.loc[r, c1] = str(sig_df.loc[r, c1]) + " " + col_letters[c2]
                    elif t_stat < -1.96:
                        sig_df.loc[r, c2] = str(sig_df.loc[r, c2]) + " " + col_letters[c1]
                except Exception:
                    continue
    return sig_df, col_letters

def apply_sig_testing(df_pct, df_n, bases=None):
    """Parami test Z (95%) dla proporcji kolumnowych w tabelach krzyzowych.

    Zwraca (sig_df, col_letters):
      sig_df     - DataFrame stringow z literami istotnych roznic
      col_letters - slownik {col: litera}
    Kolumna 'Suma' jest wykluczona z testow.

    bases: opcjonalna Series z efektywna baza (ESS) per kolumna. Gdy None,
    uzywana jest wazona liczebnosc df_n.loc['Suma']. Przy aktywnych wagach
    wywolujacy powinien podac ESS, inaczej test jest przeszacowany (zbyt wiele
    istotnych roznic) -- spojnie z apply_means_sig_testing, ktory uzywa ESS.
    """
    SKIP_ROWS = {'Suma', 'Braki danych', 'Braki danych (wykluczone z tabeli)'}
    cols = [c for c in df_pct.columns if c != 'Suma']
    letters = list(string.ascii_uppercase)
    col_letters = {c: letters[i % 26] * (i // 26 + 1) for i, c in enumerate(cols)}
    # dtype=object: pandas 2.x StringDtype odrzuca modyfikacje przez +=
    sig_df = pd.DataFrame("", index=df_pct.index, columns=df_pct.columns, dtype=object)

    if bases is None:
        try:
            bases = df_n.loc['Suma']
        except (KeyError, TypeError):
            return sig_df, col_letters

    if len(cols) < 2:
        return sig_df, col_letters

    def _flt(val):
        try:
            fv = float(val)
            return fv if np.isfinite(fv) else np.nan
        except (TypeError, ValueError):
            return np.nan

    for r in df_pct.index:
        if str(r) in SKIP_ROWS:
            continue
        for i, c1 in enumerate(cols):
            for j, c2 in enumerate(cols):
                if i >= j:
                    continue
                try:
                    raw1 = _flt(df_pct.loc[r, c1])
                    raw2 = _flt(df_pct.loc[r, c2])
                    p1 = raw1 / 100.0 if not np.isnan(raw1) else np.nan
                    p2 = raw2 / 100.0 if not np.isnan(raw2) else np.nan
                    n1 = _flt(bases[c1])
                    n2 = _flt(bases[c2])
                    if (np.isnan(p1) or np.isnan(p2)
                            or np.isnan(n1) or np.isnan(n2)
                            or n1 <= 0 or n2 <= 0):
                        continue
                    # Ogranicz proporcje do [0, 1]
                    p1 = max(0.0, min(1.0, p1))
                    p2 = max(0.0, min(1.0, p2))
                    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
                    if p_pool <= 0 or p_pool >= 1:
                        continue
                    se2 = p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2)
                    if se2 <= 0:
                        continue
                    se = np.sqrt(se2)
                    z = (p1 - p2) / se
                    if not np.isfinite(z):
                        continue
                    if z > 1.96:
                        sig_df.loc[r, c1] = str(sig_df.loc[r, c1]) + " " + col_letters[c2]
                    elif z < -1.96:
                        sig_df.loc[r, c2] = str(sig_df.loc[r, c2]) + " " + col_letters[c1]
                except Exception:
                    continue
    return sig_df, col_letters


def _banner_reorder(values_iter, order):
    """Posortuj values_iter wg listy 'order' (parametr, nie session_state).
    Wartosci spoza order dolaczane na koncu w kolejnosci wejsciowej."""
    values_list = list(values_iter)
    if not order:
        return values_list
    str_to_orig = {}
    for v in values_list:
        sv = str(v)
        if sv not in str_to_orig:
            str_to_orig[sv] = v
    seen, result = set(), []
    for o in order:
        os_ = str(o)
        if os_ in str_to_orig and os_ not in seen:
            result.append(str_to_orig[os_]); seen.add(os_)
    for v in values_list:
        sv = str(v)
        if sv not in seen:
            result.append(v); seen.add(sv)
    return result


def build_banner_table(row_var, banner_vars, df_s, df_raw_s, weights,
                       var_labels, mrs_sets=None, value_orders=None,
                       box_sets=None, measure="N+%", do_sig=False,
                       include_total=True):
    """Zbuduj tabele zbiorcza (banner): jedna zmienna w wierszach (row_var) x
    wiele zmiennych w kolumnach (banner_vars); opcjonalna kolumna 'Ogolem'
    (include_total) na KONCU tabeli.

    Zwraca (banner_df, meta):
      banner_df - DataFrame ze SPLASZCZONYMI kolumnami, np. "Plec=Kobieta [%] (A)";
                  ostatni wiersz "Baza (N)" = wazone N per kolumna.
      meta      - {etykieta_bloku: {kategoria: litera}} dla legendy (gdy do_sig).

    measure: "N+%" | "Tylko %" | "Tylko N". % zawsze kolumnowy. Test Z (do_sig)
    liczony per blok (litery A,B,C resetowane na blok), baza testu = ESS przy wagach.
    Obsluga MRS w wierszu i w kolumnach bannera (matrycowe poza zakresem).
    """
    mrs_sets     = mrs_sets or {}
    value_orders = value_orders or {}
    box_sets     = box_sets or {}
    w = (np.asarray(weights, dtype=float) if weights is not None
         else np.ones(len(df_raw_s)))
    show_n = measure in ("N+%", "Tylko N")
    show_p = measure in ("N+%", "Tylko %")
    if not show_n and not show_p:
        show_n = show_p = True

    def _mrs_cols(setdata):
        return setdata if isinstance(setdata, list) else setdata.get('cols', [])

    is_row_mrs = row_var in mrs_sets
    if is_row_mrs:
        _rcols = _mrs_cols(mrs_sets[row_var])
        row_valid = ~df_raw_s[_rcols].isna().all(axis=1).values
    else:
        row_valid = df_s[row_var].notna().values

    # Globalny wspolczynnik deflacji ESS (Sigma_w / Sigma_w^2) dla testu Z
    _sw  = float(w.sum())
    _sw2 = float((w ** 2).sum())
    _ess_ratio = (_sw / _sw2) if _sw2 > 0 else 1.0

    # Master index wierszy (spojny dla wszystkich blokow)
    if is_row_mrs:
        _rcols = _mrs_cols(mrs_sets[row_var])
        master_rows = [var_labels.get(c, c) for c in _rcols]
    else:
        _uniq = pd.Series(df_s[row_var].values).dropna().unique()
        master_rows = _banner_reorder(_uniq, value_orders.get(row_var, []))
        for _bn in box_sets.get(row_var, {}).keys():
            if _bn not in master_rows:
                master_rows.append(_bn)

    # Bloki bannera; kolumna "Ogolem" (opcjonalna) na KONCU tabeli
    blocks = [(bv, bv) for bv in banner_vars]
    if include_total:
        blocks = blocks + [("Og\u00f3\u0142em", None)]
    flat = {}            # nazwa_kolumny -> Series (indeks master_rows)
    base_row = {}        # nazwa_kolumny -> wazone N (do wiersza Baza (N))
    meta = {}

    for blk_label, col_var in blocks:
        blk_disp = var_labels.get(blk_label, blk_label) if col_var is not None else "Og\u00f3\u0142em"
        # --- df_n_block (wazone liczebnosci) + col_base (baza % per kolumna) ---
        if col_var is None:
            if is_row_mrs:
                _rcols = _mrs_cols(mrs_sets[row_var])
                _mw = df_raw_s[_rcols].replace(np.nan, 0).multiply(w, axis=0)
                _cnt = _mw.sum(axis=0)
                _cnt.index = [var_labels.get(c, c) for c in _rcols]
                df_n_block = _cnt.to_frame("Og\u00f3\u0142em")
            else:
                _tmp = pd.DataFrame({'v': df_s[row_var].values, 'w': w}).dropna(subset=['v'])
                _cnt = _tmp.groupby('v', observed=False)['w'].sum()
                df_n_block = _cnt.to_frame("Og\u00f3\u0142em")
            col_base = pd.Series({"Og\u00f3\u0142em": float(w[row_valid].sum())})
        else:
            is_col_mrs = col_var in mrs_sets
            if is_row_mrs and not is_col_mrs:
                _rcols = _mrs_cols(mrs_sets[row_var])
                _mw = df_raw_s[_rcols].replace(np.nan, 0).multiply(w, axis=0)
                df_n_block = _mw.groupby(df_s[col_var].values, observed=False).sum().T
                df_n_block.index = [var_labels.get(c, c) for c in df_n_block.index]
            elif not is_row_mrs and is_col_mrs:
                _ccols = _mrs_cols(mrs_sets[col_var])
                _mw = df_raw_s[_ccols].replace(np.nan, 0).multiply(w, axis=0)
                df_n_block = _mw.groupby(df_s[row_var].values, observed=False).sum()
                df_n_block.columns = [var_labels.get(c, c) for c in df_n_block.columns]
            elif is_row_mrs and is_col_mrs:
                # MRS x MRS: liczebnosci = wazona suma wspol-wystapien
                _rcols = _mrs_cols(mrs_sets[row_var])
                _ccols = _mrs_cols(mrs_sets[col_var])
                _rb = df_raw_s[_rcols].notna() & (df_raw_s[_rcols] != 0)
                _cb = df_raw_s[_ccols].notna() & (df_raw_s[_ccols] != 0)
                _mat = {}
                for _ci, _cc in enumerate(_ccols):
                    _csel = _cb.iloc[:, _ci].values
                    _mat[var_labels.get(_cc, _cc)] = (
                        _rb.multiply(w * _csel, axis=0).sum(axis=0).values)
                df_n_block = pd.DataFrame(_mat, index=[var_labels.get(c, c) for c in _rcols])
            else:
                df_n_block = pd.crosstab(df_s[row_var], df_s[col_var],
                                         values=w, aggfunc='sum', dropna=False).fillna(0)

            if is_col_mrs:
                col_base = df_n_block.sum(axis=0).astype(float)
            else:
                _cbdf = pd.DataFrame({'c': df_s[col_var].values, 'w': w})
                _cbdf = _cbdf[row_valid & df_s[col_var].notna().values]
                col_base = _cbdf.groupby('c', observed=False)['w'].sum().astype(float)

        # value order + box-sets na wierszach (tylko gdy wiersz nie jest MRS)
        if not is_row_mrs:
            if value_orders.get(row_var):
                df_n_block = df_n_block.reindex(
                    index=_banner_reorder(df_n_block.index, value_orders[row_var]),
                    fill_value=0)
            for _bn, _bcats in box_sets.get(row_var, {}).items():
                _present = df_n_block.index.intersection(_bcats)
                df_n_block.loc[_bn] = df_n_block.loc[_present].sum(axis=0) if len(_present) else 0

        # uspojnij do master_rows
        df_n_block = df_n_block.reindex(index=master_rows, fill_value=0)
        col_base = col_base.reindex(df_n_block.columns).fillna(0.0)

        # kolumnowy %
        _denom = col_base.replace(0, np.nan)
        df_pct_block = df_n_block.div(_denom, axis=1) * 100.0

        # suma % per kolumna (bez wierszy box-set, by nie liczyc ich podwojnie)
        # do wiersza podsumowania: kolumny [%] pokazuja sume %, kolumny [N] -> baze N
        _box_names = set(box_sets.get(row_var, {}).keys()) if not is_row_mrs else set()
        _real_rows = [r for r in df_pct_block.index if r not in _box_names]
        _pct_col_sum = df_pct_block.loc[_real_rows].sum(axis=0)

        # test Z per blok
        col_letters = {}
        sig_df = None
        if do_sig and df_n_block.shape[1] >= 2:
            try:
                _bases = col_base * _ess_ratio
                sig_df, col_letters = apply_sig_testing(df_pct_block, df_n_block, bases=_bases)
            except Exception:
                col_letters, sig_df = {}, None
        if col_letters:
            meta[blk_disp] = dict(col_letters)

        # zbuduj splaszczone kolumny
        for _cat in df_n_block.columns:
            if col_var is None:
                _name_base = "Og\u00f3\u0142em"
            else:
                _name_base = f"{blk_disp}={_cat}"
            _letter = col_letters.get(_cat, "")
            _n_ser = df_n_block[_cat]
            _p_ser = df_pct_block[_cat]
            _sig_ser = (sig_df[_cat] if (sig_df is not None and _cat in sig_df.columns)
                        else pd.Series("", index=df_n_block.index))

            if show_n:
                if not show_p and _letter:
                    # tryb Tylko N + test Z: litery doklejone do N
                    _n_name = f"{_name_base} [N] ({_letter})"
                    flat[_n_name] = pd.Series(
                        [(str(int(round(v))) + str(_sig_ser.loc[i]))
                         if pd.notna(v) else "" for i, v in _n_ser.items()],
                        index=df_n_block.index)
                else:
                    _n_name = f"{_name_base} [N]"
                    flat[_n_name] = _n_ser
                base_row[_n_name] = float(col_base.get(_cat, 0.0))
            if show_p:
                if _letter:
                    _p_name = f"{_name_base} [%] ({_letter})"
                    flat[_p_name] = pd.Series(
                        [(f"{v:.0f}%" + str(_sig_ser.loc[i]))
                         if pd.notna(v) else "" for i, v in _p_ser.items()],
                        index=df_n_block.index)
                else:
                    _p_name = f"{_name_base} [%]"
                    flat[_p_name] = _p_ser.round(0)
                base_row[_p_name] = float(_pct_col_sum.get(_cat, 0.0))

    banner_df = pd.DataFrame(flat, index=master_rows)
    # wiersz podsumowania: N dla kolumn liczebnosci, suma % dla kolumn procentowych
    banner_df.loc["Baza (N) / Suma (%)"] = pd.Series(base_row)
    return banner_df, meta


def build_banner_table_multi(row_vars, banner_vars, df_s, df_raw_s, weights,
                             var_labels, mrs_sets=None, value_orders=None,
                             box_sets=None, measure="N+%", do_sig=False,
                             include_total=True):
    """Zbuduj JEDNA tabele banner dla wielu zmiennych w wierszach (jedna pod druga).

    Kazde pytanie jest poprzedzone wierszem-naglowkiem (indeks = etykieta pytania,
    wszystkie komorki NaN) -> renderery wykrywaja taki "pusty" wiersz i pokazuja go
    jako scalony pasek na calej szerokosci. Pod naglowkiem ida kategorie pytania i
    jego wiersz "Baza (N) / Suma (%)".

    Zwraca (banner_df, meta). meta brane z pierwszego pytania, ktore ma litery --
    litery blokow sa identyczne dla wszystkich pytan (te same banner_vars => te same
    kolumny w tej samej kolejnosci => te same przypisania A,B,C). banner_df moze miec
    powtorzony indeks (np. wiele wierszy "Baza (N) / Suma (%)") -- to OK, bo renderery
    iteruja po wierszach, a serializacja JSON uzywa orient='split'.
    """
    if isinstance(row_vars, str):
        row_vars = [row_vars]
    parts = []
    meta = {}
    for rv in row_vars:
        bdf, bmeta = build_banner_table(
            rv, banner_vars, df_s, df_raw_s, weights, var_labels,
            mrs_sets=mrs_sets, value_orders=value_orders, box_sets=box_sets,
            measure=measure, do_sig=do_sig, include_total=include_total)
        if not meta and bmeta:
            meta = bmeta
        rv_label = var_labels.get(rv, rv)
        hdr = pd.DataFrame([[np.nan] * bdf.shape[1]],
                           columns=bdf.columns, index=[rv_label])
        parts.append(pd.concat([hdr, bdf], axis=0))
    banner_df = pd.concat(parts, axis=0) if parts else pd.DataFrame()
    return banner_df, meta


def parse_banner_blocks(columns):
    """Parsuje splaszczone nazwy kolumn bannera do struktury blokow ze scalonymi naglowkami.
    Zwraca list[(blok_label, list[(col_name, cat_display)])] w kolejnosci kolumn.
    Przyklad: 'Plec=Kobieta [%] (A)' -> blok='Plec', cat_display='Kobieta [%] (A)'.
              'Ogolem [%]'            -> blok='Ogolem', cat_display='[%]'.
    """
    blocks = {}
    block_order = []
    for col in columns:
        col_s = str(col)
        if '=' in col_s:
            blk, cat = col_s.split('=', 1)
        else:
            blk = col_s.split(' [')[0] if ' [' in col_s else col_s
            cat = col_s[len(blk):].lstrip() or col_s
        if blk not in blocks:
            blocks[blk] = []
            block_order.append(blk)
        blocks[blk].append((col_s, cat))
    return [(bl, blocks[bl]) for bl in block_order]


def module_header(icon, title, subtitle=""):
    """Render a blue gradient banner header identical to the Dashboard banner."""
    # Back button (ukryty dla admina ktory nie ma dostepu do Dashboardu)
    if st.session_state.get("current_user_role") != "admin":
        if st.button("\u2190 Powr\u00f3\u0107 do Dashboardu", key=f"back_dash_{title}"):
            st.session_state.nav_to = "\U0001f3e0 Dashboard"
            st.rerun()

    sub_html = (f'<p style="margin:6px 0 0;opacity:.85;font-size:0.95rem;">{subtitle}</p>'
                if subtitle else "")
    st.markdown(f"""
<div style="background:linear-gradient(90deg,#1F4E79,#2E75B6);
     padding:22px 32px;border-radius:10px;margin-bottom:20px;color:white;">
  <h2 style="margin:0;font-size:1.55rem;">{icon} {title}</h2>
  {sub_html}
</div>
""", unsafe_allow_html=True)

    # Show active split indicator below the header
    _sv = st.session_state.get('split_var')
    if _sv:
        st.markdown(
            f'<div style="background:#FFF4CE;border-left:4px solid #D97706;'
            f'padding:10px 16px;margin-bottom:15px;border-radius:4px;">'
            f'<strong>\U0001f500 Aktywny podzia\u0142 na podzbiory:</strong> '
            f'<code>{_sv}</code> \u2014 wyniki b\u0119d\u0105 liczone osobno dla ka\u017cdej grupy. '
            f'<small>(zmie\u0144 w <em>Przygotowanie Danych \u2192 Podzia\u0142 na podzbiory</em>)</small>'
            f'</div>',
            unsafe_allow_html=True
        )

    # Show active weights indicator below the header
    _w = st.session_state.get('weights')
    if _w is not None:
        try:
            _n_w = len(_w)
            _mn  = float(np.min(_w))
            _mx  = float(np.max(_w))
            _targets = st.session_state.get('weight_targets', {}) or {}
            _vars = ", ".join(f"<code>{v}</code>" for v in _targets.keys()) if _targets else "<em>custom</em>"
        except Exception:
            _n_w = "?"; _mn = 0; _mx = 0; _vars = ""
        st.markdown(
            f'<div style="background:#E2F0D9;border-left:4px solid #548235;'
            f'padding:10px 16px;margin-bottom:15px;border-radius:4px;">'
            f'<strong>\u2696\ufe0f Aktywne wagi:</strong> '
            f'N = {_n_w:,} \u00b7 min = {_mn:.3f} \u00b7 max = {_mx:.3f} \u00b7 '
            f'zmienne: {_vars} '
            f'<small>(zarz\u0105dzaj w <em>Przygotowanie Danych \u2192 Wa\u017cenie</em>)</small>'
            f'</div>',
            unsafe_allow_html=True
        )


def _format_means_table(df):
    """Format means-style table (index-based) with row-aware logic:
       - 'Baza (N)' or 'N' row \u2192 integer (rounded for weighted N)
       - Other numeric rows \u2192 2 decimal places"""
    out = df.copy().astype(object)
    for r_idx in out.index:
        is_n_row = ("baza (n)" in str(r_idx).lower()
                    or str(r_idx).strip().lower() == "n"
                    or "liczebno" in str(r_idx).lower())
        for c_idx in out.columns:
            v = out.loc[r_idx, c_idx]
            if pd.isna(v) or isinstance(v, str):
                continue
            try:
                fv = float(v)
                if is_n_row:
                    out.loc[r_idx, c_idx] = f"{int(round(fv)):,}"
                else:
                    out.loc[r_idx, c_idx] = f"{fv:.2f}"
            except (ValueError, TypeError):
                pass
    return out

def get_streamlit_format(df):
    format_dict = {}
    # Keywords that indicate a count/N column (always integer)
    _n_keywords = ("[n]", "liczebno", "liczba", " n ", "_n_", "n_valid",
                   "n_pair", "df", "stopni swobody", "observ")
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, str) and ("%" in x or any(c.isalpha() for c in x if c not in ['N', 'a', 'A']))).any(): continue
        col_str = str(col)
        col_low = col_str.lower()
        is_n_col = (
            col_str in ("N", "Liczebnosc [N]", "Liczebno\u015b\u0107 [N]")
            or "[N]" in col_str
            or any(kw in col_low for kw in _n_keywords)
            or col_low.startswith("n ") or col_low.endswith(" n") or col_low == "n"
        )
        if "%" in col_low or "procent" in col_low:
            format_dict[col] = lambda x: f"{x:.1f}%" if pd.notnull(x) and not isinstance(x, str) else str(x)
        elif is_n_col:
            format_dict[col] = lambda x: f"{round(x):.0f}" if pd.notnull(x) and not isinstance(x, str) else str(x)
        else:
            format_dict[col] = lambda x: f"{x:.2f}" if pd.notnull(x) and not isinstance(x, str) and abs(x - round(x)) > 0.01 else (f"{x:.0f}" if pd.notnull(x) and not isinstance(x, str) else str(x))
    return format_dict

def safe_style(df):
    """
    Apply get_streamlit_format styling safely.
    Pandas Styler crashes on non-unique index or columns \u2014 deduplicate before styling.
    """
    d = df.copy()
    # Deduplicate index
    if not d.index.is_unique:
        seen = {}
        new_idx = []
        for v in d.index:
            k = str(v)
            if k in seen:
                seen[k] += 1
                new_idx.append(f"{v} ({seen[k]})")
            else:
                seen[k] = 0
                new_idx.append(v)
        d.index = new_idx
    # Deduplicate columns
    if not d.columns.is_unique:
        seen = {}
        new_cols = []
        for v in d.columns:
            k = str(v)
            if k in seen:
                seen[k] += 1
                new_cols.append(f"{v} ({seen[k]})")
            else:
                seen[k] = 0
                new_cols.append(v)
        d.columns = new_cols
    return d.style.format(get_streamlit_format(d))

# ---------------------------------------------------------------------------
# Module-level style/format helpers (defined once, reused across all reruns)
# ---------------------------------------------------------------------------

def _to_float_pct(x):
    """Strip % and letter suffixes, return float or original value."""
    if isinstance(x, str):
        clean = x.replace('%', '').strip().split()[0] if x.strip() else ''
        try:
            return float(clean)
        except (ValueError, TypeError):
            return x
    return x

def _fmt_cell(v):
    """Format a matrix table cell for display."""
    if v == "" or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, float):
        return f"{v:.1f}"
    return str(v)

def _style_p(val):
    """Color p-value cells: green if significant, red if not."""
    try:
        v = float(val)
        if v < 0.001: return 'color:#006100;font-weight:bold'
        if v < 0.05:  return 'color:#006100'
        return 'color:#C00000'
    except (ValueError, TypeError):
        return ''

def _style_vif(val):
    """Color VIF cells: red > 10, orange > 5, green otherwise."""
    try:
        v = float(val)
        if v > 10: return 'color:#C00000;font-weight:bold'
        if v > 5:  return 'color:#E36C09'
        return 'color:#006100'
    except (ValueError, TypeError):
        return ''

def _style_loading(val):
    """Bold + green background for factor loadings >= 0.4."""
    try:
        if abs(float(val)) >= 0.4:
            return 'font-weight:bold;background-color:#E2EFDA'
        return ''
    except (ValueError, TypeError):
        return ''

def _color_pair_row(row):
    """Color correlation pair rows by strength."""
    abs_r = abs(row['r'])
    if abs_r >= 0.7:
        c = '#E2EFDA' if row['r'] > 0 else '#FCE4D6'
    else:
        c = '#FFFACD'
    return [f'background-color: {c}'] * len(row)

def _color_sig(row):
    """Green background for significant logistic regression rows."""
    color = '#E2EFDA' if row.get('Istotny') == 'Tak' else ''
    return [f'background-color: {color}'] * len(row)

def _make_style_matrix_row(sumrow_label):
    """Factory: returns a styler for matrix table rows."""
    def _style(row):
        if row.name == sumrow_label:
            return ['background-color:#E2EFDA; font-weight:bold'] * len(row)
        return [''] * len(row)
    return _style

def _make_color_corr_cell(threshold):
    """Factory: returns a cell styler for correlation matrix given threshold."""
    def _style(val):
        try:
            v = float(str(val).split()[0])
        except (ValueError, TypeError):
            return ''
        if abs(v) >= 1.0:
            return ''
        abs_v = abs(v)
        if abs_v >= 0.7:
            bg = '#C00000' if v < 0 else '#375623'
            return f'background-color: {bg}; color: white; font-weight: bold'
        elif abs_v >= threshold:
            bg = '#FCE4D6' if v < 0 else '#E2EFDA'
            return f'background-color: {bg}; font-weight: bold'
        return ''
    return _style

def _make_style_md(n_rows):
    """Factory: returns a MaxDiff row styler knowing total row count."""
    def _style(row):
        if row['Ranking'] == 1:
            return ['background-color:#E2EFDA; font-weight:bold'] * len(row)
        if row['Ranking'] == n_rows:
            return ['background-color:#FCE4D6'] * len(row)
        return [''] * len(row)
    return _style

def calculate_rim_weights(df, target_dict, max_iterations=50):
    n = len(df)
    weights = np.ones(n)
    for iteration in range(max_iterations):
        max_error = 0
        for var, targets in target_dict.items():
            for cat, target_pct in targets.items():
                mask = (df[var] == cat)
                if mask.sum() == 0: continue
                current_pct = weights[mask].sum() / weights.sum()
                if current_pct > 0:
                    adjustment = target_pct / current_pct
                    weights[mask] *= adjustment
                    max_error = max(max_error, abs(target_pct - current_pct))
        if max_error < 0.001: break
    # Normalize so that sum(weights) == N (SPSS default behavior)
    if weights.sum() > 0:
        weights = weights * (n / weights.sum())
    return weights

def calculate_correlations(df, cols, weights=None, method='pearson'):
    """
    Compute correlation matrix with optional case weights.
    For Pearson: uses weighted covariance matrix (consistent with SPSS WLS).
    For Spearman/Kendall: weights are approximated via replication (SPSS approach).
    Returns (corr_matrix_with_stars, n_effective).
    """
    df_clean = df[cols].dropna()

    if weights is not None:
        w = pd.Series(weights, index=df.index).reindex(df_clean.index).fillna(0)
        w = w.clip(lower=0)
    else:
        w = pd.Series(np.ones(len(df_clean)), index=df_clean.index)

    n = int(w.sum())
    # Efektywna wielkosc proby (ESS) do testu istotnosci przy wagach; n
    # (zwracane/wyswietlane) pozostaje liczebnoscia wazna. Bez wag ESS = n.
    _w2sum = float((w.values ** 2).sum())
    n_eff = int(round((float(w.sum()) ** 2) / _w2sum)) if _w2sum > 0 else n
    corr_matrix = pd.DataFrame(index=cols, columns=cols)

    if method == 'pearson':
        # Weighted Pearson: r = cov_w(x,y) / (sd_w(x) * sd_w(y))
        w_sum = w.sum()
        w_arr = w.values

        def _wcov(x, y, w_arr, w_sum):
            mx = (x * w_arr).sum() / w_sum
            my = (y * w_arr).sum() / w_sum
            return (w_arr * (x - mx) * (y - my)).sum() / w_sum

        for c1 in cols:
            for c2 in cols:
                if c1 == c2:
                    corr_matrix.loc[c1, c2] = "1.000"
                else:
                    try:
                        x = df_clean[c1].values.astype(float)
                        y = df_clean[c2].values.astype(float)
                        cov_xy = _wcov(x, y, w_arr, w_sum)
                        cov_xx = _wcov(x, x, w_arr, w_sum)
                        cov_yy = _wcov(y, y, w_arr, w_sum)
                        denom = np.sqrt(cov_xx * cov_yy)
                        r = cov_xy / denom if denom > 0 else 0.0
                        r = max(-1.0, min(1.0, r))
                        # t-stat for significance (ESS, nie wazone N -> brak przeszacowania)
                        df_t = max(n_eff - 2, 1)
                        t = r * np.sqrt(df_t / max(1 - r**2, 1e-12))
                        p = 2 * stats.t.sf(abs(t), df_t)
                        stars = "**" if p < 0.01 else ("*" if p < 0.05 else "")
                        corr_matrix.loc[c1, c2] = f"{r:.3f}{stars}\n(p={p:.3f})"
                    except Exception:
                        corr_matrix.loc[c1, c2] = "N/A"
    else:
        # Spearman / Kendall \u2014 use unweighted (methodologically standard;
        # SPSS does not weight rank-based correlations either)
        df_c2 = df_clean.copy()
        for c1 in cols:
            for c2 in cols:
                if c1 == c2:
                    corr_matrix.loc[c1, c2] = "1.000"
                else:
                    try:
                        if method == 'spearman':
                            r, p = stats.spearmanr(df_c2[c1], df_c2[c2])
                        else:
                            r, p = stats.kendalltau(df_c2[c1], df_c2[c2])
                        stars = "**" if p < 0.01 else ("*" if p < 0.05 else "")
                        corr_matrix.loc[c1, c2] = f"{r:.3f}{stars}\n(p={p:.3f})"
                    except Exception:
                        corr_matrix.loc[c1, c2] = "N/A"

    return corr_matrix, n

# -------------------------------------------------------------
# REGRESJA
# -------------------------------------------------------------

def run_regression_block(df_data, dep_var, indep_vars_blocks, weights=None):
    results_list = []
    prev_r2 = 0.0
    cumulative_vars = []
    for block_idx, block_vars in enumerate(indep_vars_blocks):
        cumulative_vars = cumulative_vars + block_vars
        df_reg = df_data[[dep_var] + cumulative_vars].dropna()
        if len(df_reg) < len(cumulative_vars) + 2:
            results_list.append({'error': f"Blok {block_idx + 1}: Za ma\u0142o obserwacji ({len(df_reg)})."})
            continue
        y = df_reg[dep_var]
        X = sm.add_constant(df_reg[cumulative_vars])
        try:
            if weights is not None:
                w_reg = pd.Series(weights, index=df_data.index).reindex(df_reg.index).fillna(1).clip(lower=0)
                model = sm.WLS(y, X, weights=w_reg).fit()
            else:
                model = sm.OLS(y, X).fit()
        except Exception as e:
            results_list.append({'error': str(e)})
            continue
        r2 = model.rsquared
        r2_adj = model.rsquared_adj
        delta_r2 = r2 - prev_r2
        f_stat = model.fvalue
        f_pval = model.f_pvalue
        if block_idx == 0:
            f_change, f_change_p, df1_change = f_stat, f_pval, len(cumulative_vars)
        else:
            df1_change = len(block_vars)
            df2_change = len(df_reg) - len(cumulative_vars) - 1
            if df2_change > 0 and (1 - r2) > 0:
                f_change = (delta_r2 / df1_change) / ((1 - r2) / df2_change)
                f_change_p = 1 - stats.f.cdf(f_change, df1_change, df2_change)
            else:
                f_change, f_change_p = np.nan, np.nan
        vif_dict = {}
        if len(cumulative_vars) > 1:
            X_vif = df_reg[cumulative_vars].astype(float)
            for i, v in enumerate(cumulative_vars):
                try:
                    vif_dict[v] = variance_inflation_factor(X_vif.values, i)
                except:
                    vif_dict[v] = np.nan
        else:
            vif_dict[cumulative_vars[0]] = np.nan
        std_y = y.std()
        beta_dict = {}
        for v in cumulative_vars:
            std_x = df_reg[v].std()
            beta_dict[v] = model.params[v] * std_x / std_y if std_y > 0 and std_x > 0 else np.nan
        coef_rows = []
        for v in cumulative_vars:
            vif_val = vif_dict.get(v, np.nan)
            coef_rows.append({
                'Zmienna': v,
                'B': model.params.get(v, np.nan),
                'B\u0142\u0105d std. B': model.bse.get(v, np.nan),
                'Beta (std.)': beta_dict.get(v, np.nan),
                't': model.tvalues.get(v, np.nan),
                'p-value': model.pvalues.get(v, np.nan),
                'VIF': vif_val,
                'Tolerancja': 1 / vif_val if pd.notna(vif_val) and vif_val > 0 else np.nan,
            })
        results_list.append({
            'Blok': block_idx + 1,
            'Zmienne w bloku': ', '.join(block_vars),
            'Wszystkie predyktory': cumulative_vars[:],
            'dep_var': dep_var,
            'N': len(df_reg),
            'R': np.sqrt(r2),
            'R2': r2,
            'Skor_R2': r2_adj,
            'Delta_R2': delta_r2,
            'F modelu': f_stat,
            'p (F modelu)': f_pval,
            'F zmiany': f_change,
            'p (F zmiany)': f_change_p,
            'df1 (F zmiany)': df1_change,
            'df2 (F zmiany)': len(df_reg) - len(cumulative_vars) - 1,
            'coef_df': pd.DataFrame(coef_rows),
            '_model': model,
            '_df_reg': df_reg,
        })
        prev_r2 = r2
    return results_list

# -------------------------------------------------------------
# ANOVA
# -------------------------------------------------------------

def run_anova(df_raw, dep_var, group_var, df_labeled, weights=None):
    """One-way ANOVA with post-hoc Tukey HSD. Supports case weights (SPSS-compatible)."""
    tmp = pd.DataFrame({
        'dep': df_raw[dep_var].values,
        'grp': df_labeled[group_var].values
    }, index=df_raw.index)
    if weights is not None:
        tmp['w'] = pd.Series(weights, index=df_raw.index).reindex(tmp.index).fillna(0).clip(lower=0)
    else:
        tmp['w'] = 1.0
    tmp = tmp.dropna(subset=['dep', 'grp'])
    tmp = tmp[tmp['w'] > 0]

    groups = tmp['grp'].unique()
    group_data = [tmp.loc[tmp['grp'] == g, 'dep'].values for g in groups]
    group_w    = [tmp.loc[tmp['grp'] == g, 'w'].values  for g in groups]
    group_data = [(d, w) for d, w in zip(group_data, group_w) if len(d) >= 2]
    if len(group_data) < 2:
        return None, "Za ma\u0142o grup z wystarczaj\u0105c\u0105 liczb\u0105 obserwacji."

    # Weighted grand mean
    total_w  = tmp['w'].sum()
    grand_mean = (tmp['dep'] * tmp['w']).sum() / total_w

    # Weighted group stats
    desc_rows = []
    for g in groups:
        sub = tmp[tmp['grp'] == g]
        n_g   = sub['w'].sum()
        mean_g = (sub['dep'] * sub['w']).sum() / n_g if n_g > 0 else np.nan
        var_g  = (sub['w'] * (sub['dep'] - mean_g) ** 2).sum() / max(n_g - 1, 1)
        std_g  = np.sqrt(var_g)
        desc_rows.append({
            'Grupa': g,
            'N (wa\u017cone)': int(round(n_g)),
            'Srednia': round(mean_g, 4),
            'Odch. std.': round(std_g, 4),
            'Min': sub['dep'].min(),
            'Max': sub['dep'].max()
        })
    desc_df = pd.DataFrame(desc_rows)

    # Weighted SS
    ss_between = sum(
        (tmp[tmp['grp'] == g]['w'].sum()) *
        ((tmp[tmp['grp'] == g]['dep'] * tmp[tmp['grp'] == g]['w']).sum() /
         tmp[tmp['grp'] == g]['w'].sum() - grand_mean) ** 2
        for g in groups
    )
    ss_total   = (tmp['w'] * (tmp['dep'] - grand_mean) ** 2).sum()
    ss_within  = ss_total - ss_between

    df_between = len(groups) - 1
    df_within  = total_w - len(groups)   # effective df (sum of weights - k)
    ms_between = ss_between / df_between if df_between > 0 else np.nan
    ms_within  = ss_within  / df_within  if df_within  > 0 else np.nan
    f_stat     = ms_between / ms_within  if ms_within and ms_within > 0 else np.nan
    p_val      = stats.f.sf(f_stat, df_between, df_within) if not np.isnan(f_stat) else np.nan

    # Levene's test (unweighted \u2014 Levene is robust; SPSS uses unweighted too)
    lev_stat, lev_p = stats.levene(*[d for d, _ in group_data])

    eta2 = ss_between / ss_total if ss_total > 0 else np.nan

    # Tukey HSD post-hoc (weighted)
    from itertools import combinations
    posthoc_rows = []
    for g1, g2 in combinations(groups, 2):
        s1 = tmp[tmp['grp'] == g1]; s2 = tmp[tmp['grp'] == g2]
        n1  = s1['w'].sum(); n2  = s2['w'].sum()
        m1  = (s1['dep'] * s1['w']).sum() / n1 if n1 > 0 else np.nan
        m2  = (s2['dep'] * s2['w']).sum() / n2 if n2 > 0 else np.nan
        if n1 < 2 or n2 < 2 or pd.isna(ms_within): continue
        diff    = m1 - m2
        se_tukey = np.sqrt(ms_within * (1 / n1 + 1 / n2) / 2)
        q = abs(diff) / se_tukey if se_tukey > 0 else np.nan
        try:
            from scipy.stats import studentized_range
            p_tukey = studentized_range.sf(q * np.sqrt(2), len(groups), df_within)
        except Exception:
            p_tukey = np.nan
        posthoc_rows.append({
            'Grupa A': g1, 'Grupa B': g2,
            'R\u00f3\u017cnica \u015brednich (A-B)': round(diff, 4),
            'p-value (Tukey)': round(p_tukey, 4) if not np.isnan(p_tukey) else np.nan,
            'Istotna (p<0.05)': '\u2705' if (not np.isnan(p_tukey) and p_tukey < 0.05) else '\u274c'
        })

    posthoc_df = pd.DataFrame(posthoc_rows) if posthoc_rows else pd.DataFrame()

    result = {
        'dep_var': dep_var, 'group_var': group_var,
        'F': f_stat, 'p': p_val,
        'df_between': df_between, 'df_within': df_within,
        'eta2': eta2, 'lev_stat': lev_stat, 'lev_p': lev_p,
        'desc_df': desc_df, 'posthoc_df': posthoc_df,
        'ss_between': ss_between, 'ss_within': ss_within, 'ss_total': ss_total,
        'ms_between': ms_between, 'ms_within': ms_within,
    }
    return result, None

# -------------------------------------------------------------
# ANALIZA CZYNNIKOWA
# -------------------------------------------------------------

def run_factor_analysis(df_raw, cols, n_factors, rotation='varimax', method='principal', weights=None):
    df_fa = df_raw[cols].dropna()
    if len(df_fa) < len(cols) + 5:
        return None, f"Za ma\u0142o obserwacji ({len(df_fa)}). Potrzeba co najmniej {len(cols)+5}."
    if n_factors >= len(cols):
        return None, f"Liczba czynnik\u00f3w ({n_factors}) musi by\u0107 mniejsza ni\u017c liczba zmiennych ({len(cols)})."
    try:
        # When weights are provided, build weighted correlation matrix and pass to FA.
        # This is consistent with SPSS FACTOR (WLS/GLS approach via cov_matrix).
        if weights is not None:
            w = pd.Series(weights, index=df_raw.index).reindex(df_fa.index).fillna(0).clip(lower=0)
            w_arr = w.values
            w_sum = w_arr.sum()
            X = df_fa.values.astype(float)
            # Weighted means
            means = (w_arr[:, None] * X).sum(axis=0) / w_sum
            Xc = X - means
            # Weighted covariance matrix
            cov_w = (w_arr[:, None] * Xc).T @ Xc / (w_sum - 1)
            # Convert to correlation matrix
            std_w = np.sqrt(np.diag(cov_w))
            corr_w = cov_w / np.outer(std_w, std_w)
            np.fill_diagonal(corr_w, 1.0)
            corr_df_w = pd.DataFrame(corr_w, index=cols, columns=cols)
            # KMO and Bartlett on weighted corr
            try:
                kmo_all, kmo_model = calculate_kmo(corr_df_w)
                bart_chi2, bart_p = calculate_bartlett_sphericity(corr_df_w)
            except Exception:
                kmo_all, kmo_model = calculate_kmo(df_fa)
                bart_chi2, bart_p = calculate_bartlett_sphericity(df_fa)
            fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation, method=method,
                                is_corr_matrix=True)
            fa.fit(corr_df_w)
        else:
            kmo_all, kmo_model = calculate_kmo(df_fa)
            bart_chi2, bart_p = calculate_bartlett_sphericity(df_fa)
            fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation, method=method)
            fa.fit(df_fa)

        loadings = pd.DataFrame(fa.loadings_, index=cols,
                                columns=[f"Czynnik {i + 1}" for i in range(n_factors)])
        communalities = pd.DataFrame({'Komunalnosc (h2)': fa.get_communalities()}, index=cols)
        ev, v = fa.get_eigenvalues()
        eigenvalues_df = pd.DataFrame({'Warto\u015b\u0107 w\u0142asna': ev, '% wariancji': ev / len(cols) * 100,
                                       'Skumulowany %': np.cumsum(ev / len(cols) * 100)},
                                      index=[f"Czynnik {i + 1}" for i in range(len(ev))])
        var_explained = fa.get_factor_variance()
        var_df = pd.DataFrame({'SS \u0141adunk\u00f3w': var_explained[0], '% wariancji': var_explained[1] * 100,
                               'Skumulowany %': var_explained[2] * 100},
                              index=[f"Czynnik {i + 1}" for i in range(n_factors)])
        return {
            'loadings': loadings, 'communalities': communalities,
            'eigenvalues': eigenvalues_df, 'variance': var_df,
            'kmo': kmo_model, 'kmo_all': kmo_all,
            'bartlett_chi2': bart_chi2, 'bartlett_p': bart_p,
            'n': len(df_fa), 'cols': cols, 'rotation': rotation,
        }, None
    except Exception as e:
        return None, str(e)

# -------------------------------------------------------------
# EKSPORT DO EXCELA -- TABLICE WYNIKOWE (NAPRAWIONY)
# -------------------------------------------------------------

def safe_excel_val(val):
    """Convert value to Excel-safe type."""
    if val is None:
        return ""
    if isinstance(val, float) and np.isnan(val):
        return ""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, pd.CategoricalDtype):
        return str(val)
    return val

def export_toc_sheet(writer, results, matrix_results, var_labels, sheet_map,
                     regression_results=None, anova_results=None, factor_results=None,
                     conjoint_results=None, maxdiff_results=None,
                     pre_created_ws=None):
    """
    Create a 'Spis Tre\u015bci' sheet with clickable hyperlinks to each table.
    If pre_created_ws is provided (worksheet created before data sheets), use it directly
    so the ToC appears as the first tab in Excel.
    sheet_map: dict {sheet_name: {title: excel_row}} -- row index where each table starts.
    """
    workbook = writer.book
    if pre_created_ws is not None:
        worksheet = pre_created_ws
    else:
        worksheet = workbook.add_worksheet('Spis Tre\u015bci')
        worksheet.set_tab_color('#1F4E79')
    worksheet.activate()

    fmt_title   = workbook.add_format({'bold': True, 'font_size': 14, 'font_color': '#1F4E79',
                                        'bottom': 2, 'bottom_color': '#1F4E79'})
    fmt_section = workbook.add_format({'bold': True, 'bg_color': '#1F4E79', 'font_color': 'white',
                                        'border': 1, 'align': 'left', 'font_size': 11})
    fmt_link    = workbook.add_format({'font_color': '#0563C1', 'underline': True, 'border': 1, 'align': 'left'})
    fmt_sub     = workbook.add_format({'italic': True, 'font_color': '#595959', 'border': 1, 'align': 'left'})
    fmt_hdr     = workbook.add_format({'bold': True, 'bg_color': '#D6E4F0', 'border': 1})
    fmt_num     = workbook.add_format({'border': 1, 'align': 'right', 'num_format': '#,##0'})
    fmt_empty   = workbook.add_format({'border': 1})

    worksheet.set_column(0, 0,  6)   # #
    worksheet.set_column(1, 1, 55)   # Nazwa tabeli
    worksheet.set_column(2, 2, 22)   # Arkusz
    worksheet.set_column(3, 3, 12)   # Typ

    row = 0
    worksheet.merge_range(row, 0, row, 3, 'Spis Tre\u015bci Raportu Analitycznego', fmt_title)
    row += 2

    counter = 1

    # -- Tablice cz?sto?ci --
    czestosci = results.get('czestosci', {})
    if czestosci:
        worksheet.merge_range(row, 0, row, 3, 'Tablice Cz\u0119sto\u015bci', fmt_section)
        row += 1
        worksheet.write(row, 0, '#',          fmt_hdr)
        worksheet.write(row, 1, 'Zmienna',    fmt_hdr)
        worksheet.write(row, 2, 'Arkusz',     fmt_hdr)
        worksheet.write(row, 3, 'Typ',        fmt_hdr)
        row += 1
        for title in czestosci:
            display = f"[{title}] {var_labels.get(title, title)}"
            target_row = sheet_map.get('Cz\u0119sto\u015bci', {}).get(title, 1)
            cell_addr = f"A{target_row + 1}"
            try:
                worksheet.write_url(row, 1, f"internal:'Cz\u0119sto\u015bci'!{cell_addr}", fmt_link, display)
            except Exception:
                worksheet.write(row, 1, display, fmt_link)
            worksheet.write(row, 0, counter,     fmt_num)
            worksheet.write(row, 2, 'Cz\u0119sto\u015bci', fmt_empty)
            worksheet.write(row, 3, 'Cz\u0119sto\u015bci', fmt_empty)
            counter += 1
            row += 1
        row += 1

    # -- Tablice krzy?owe --
    krzyzowe = results.get('krzyzowe', {})
    if krzyzowe:
        worksheet.merge_range(row, 0, row, 3, 'Tablice Krzy\u017cowe', fmt_section)
        row += 1
        worksheet.write(row, 0, '#',         fmt_hdr)
        worksheet.write(row, 1, 'Tabela',    fmt_hdr)
        worksheet.write(row, 2, 'Arkusz',    fmt_hdr)
        worksheet.write(row, 3, 'Typ',       fmt_hdr)
        row += 1
        for title in krzyzowe:
            if ' x ' in title:
                rv, cv = title.split(' x ', 1)
                display = (f"Wiersz: [{rv}] {var_labels.get(rv, rv)}  \u00d7  "
                           f"Kolumna: [{cv}] {var_labels.get(cv, cv)}")
            else:
                display = title
            target_row = sheet_map.get('Krzy\u017cowe', {}).get(title, 1)
            cell_addr = f"A{target_row + 1}"
            try:
                worksheet.write_url(row, 1, f"internal:'Krzy\u017cowe'!{cell_addr}", fmt_link, display)
            except Exception:
                worksheet.write(row, 1, display, fmt_link)
            worksheet.write(row, 0, counter,    fmt_num)
            worksheet.write(row, 2, 'Krzy\u017cowe', fmt_empty)
            worksheet.write(row, 3, 'Krzy\u017cowe', fmt_empty)
            counter += 1
            row += 1
        row += 1

    # -- Pytania matrycowe --
    if matrix_results:
        worksheet.merge_range(row, 0, row, 3, 'Pytania Matrycowe', fmt_section)
        row += 1
        worksheet.write(row, 0, '#',       fmt_hdr)
        worksheet.write(row, 1, 'Pytanie', fmt_hdr)
        worksheet.write(row, 2, 'Arkusz',  fmt_hdr)
        worksheet.write(row, 3, 'Typ',     fmt_hdr)
        row += 1
        for entry in matrix_results:
            try:
                worksheet.write_url(row, 1, "internal:'Pytania Matrycowe'!A1", fmt_link, entry['name'])
            except Exception:
                worksheet.write(row, 1, entry['name'], fmt_link)
            worksheet.write(row, 0, counter,              fmt_num)
            worksheet.write(row, 2, 'Pytania Matrycowe',  fmt_empty)
            worksheet.write(row, 3, 'Matryca',            fmt_empty)
            counter += 1
            row += 1
        row += 1

    # -- Other sheets --
    other_sheets = [('\u015arednie', 'srednie'), ('Opisowe', 'opisowe'),
                    ('Korelacje', 'korelacje')]
    for sheet_name, key in other_sheets:
        if results.get(key):
            worksheet.write(row, 0, '', fmt_empty)
            try:
                worksheet.write_url(row, 1, f"internal:'{sheet_name}'!A1", fmt_link, sheet_name)
            except Exception:
                worksheet.write(row, 1, sheet_name, fmt_link)
            worksheet.write(row, 2, sheet_name, fmt_empty)
            worksheet.write(row, 3, '',          fmt_empty)
            row += 1

    # -- Optional analytical sheets (only shown if results exist) --
    valid_reg = [r for r in (regression_results or []) if 'error' not in r]
    has_anova  = bool(anova_results)
    has_fa     = bool(factor_results)
    valid_conj = [r for r in (conjoint_results or []) if not r.get('error')]
    has_md     = bool(maxdiff_results)

    for sheet_name, has_data in [
        ('Regresja',         bool(valid_reg)),
        ('ANOVA',            has_anova),
        ('Anal. Czynnikowa', has_fa),
        ('Conjoint',         bool(valid_conj)),
        ('MaxDiff',          has_md),
    ]:
        if not has_data:
            continue
        try:
            worksheet.write_url(row, 1, f"internal:'{sheet_name}'!A1", fmt_link, sheet_name)
            worksheet.write(row, 0, '', fmt_empty)
            worksheet.write(row, 2, sheet_name, fmt_empty)
            worksheet.write(row, 3, '', fmt_empty)
            row += 1
        except Exception:
            pass


def export_tables_to_sheet(writer, s_name, results_dict, var_labels, add_charts=False):
    workbook = writer.book

    # Limit sheet name to 31 chars (Excel limit)
    sheet_name = s_name[:31]
    worksheet = workbook.add_worksheet(sheet_name)

    fmt_title     = workbook.add_format({'bold': True, 'align': 'center', 'valign': 'vcenter', 'border': 1, 'bg_color': '#1F4E79', 'font_color': 'white', 'font_size': 10})
    fmt_header    = workbook.add_format({'bold': True, 'align': 'center', 'valign': 'vcenter', 'border': 1, 'bg_color': '#D6E4F0', 'text_wrap': True})
    fmt_index_b   = workbook.add_format({'bold': True, 'border': 1, 'align': 'left', 'bg_color': '#F2F2F2'})
    fmt_index_n   = workbook.add_format({'border': 1, 'align': 'left'})
    fmt_n         = workbook.add_format({'num_format': '#,##0', 'border': 1, 'align': 'right'})
    fmt_pct       = workbook.add_format({'num_format': '0.0"%"', 'border': 1, 'align': 'right'})
    fmt_float     = workbook.add_format({'num_format': '#,##0.00', 'border': 1, 'align': 'right'})
    fmt_str       = workbook.add_format({'border': 1, 'align': 'center', 'text_wrap': True})
    fmt_empty     = workbook.add_format({})
    fmt_qhdr      = workbook.add_format({'bold': True, 'align': 'left', 'valign': 'vcenter', 'border': 1, 'bg_color': '#2E75B6', 'font_color': 'white', 'font_size': 10})

    worksheet.set_column(0, 0, 42)

    sr = 0
    title_row_map = {}   # title -> excel row number (for ToC hyperlinks)

    # Summary rows that should be excluded from charts
    _chart_exclude = {
        'Suma', 'Braki danych', 'Braki danych (wykluczone z tabeli)',
        'Og\u00f3\u0142em (Wa\u017cne)', 'Baza (N) / Suma (%)',
    }

    for title, df_res in results_dict.items():
        title_row_map[title] = sr   # record where this table starts
        df_export = df_res.copy()

        # Split off group label (format: "base_title | group_label")
        _grp_suffix = ""
        _base_title = title
        if " | " in title and "=" in title.split(" | ", 1)[-1]:
            _base_title, _grp_suffix = title.rsplit(" | ", 1)

        # Convert percentage columns to float for proper formatting
        if s_name != 'Korelacje':
            for col in df_export.columns:
                col_str = str(col).lower()
                if "%" in col_str or "procent" in col_str:
                    df_export[col] = df_export[col].apply(_to_float_pct)

        num_cols = len(df_export.columns)

        # Build display title
        if s_name == 'Cz\u0119sto\u015bci':
            display_title = f"[{_base_title}] {var_labels.get(_base_title, _base_title)}"
            chart_title   = var_labels.get(_base_title, _base_title)
        elif s_name in ['Krzy\u017cowe', '\u015arednie']:
            if ' x ' in _base_title:
                r_v, c_v = _base_title.split(' x ', 1)
                display_title = f"Wiersz: [{r_v}] {var_labels.get(r_v, r_v)}  \u00d7  Kolumna: [{c_v}] {var_labels.get(c_v, c_v)}"
            else:
                display_title = _base_title
            chart_title = display_title
        else:
            display_title = _base_title
            chart_title   = _base_title

        # Append group label to titles if present
        if _grp_suffix:
            display_title = f"{display_title}  \u2014  \U0001f500 {_grp_suffix}"
            chart_title   = f"{chart_title} \u2014 {_grp_suffix}"

        # Title row
        if num_cols > 1:
            worksheet.merge_range(sr, 0, sr, num_cols, display_title, fmt_title)
        else:
            worksheet.write(sr, 0, display_title, fmt_title)
        sr += 1

        # Header row(s)
        if s_name == 'Banner':
            _bn_blks = parse_banner_blocks(df_export.columns)
            fmt_blk_hdr = workbook.add_format({
                'bold': True, 'align': 'center', 'valign': 'vcenter', 'border': 1,
                'bg_color': '#1F4E79', 'font_color': 'white', 'font_size': 10,
                'text_wrap': True})
            fmt_cat_hdr = workbook.add_format({
                'bold': True, 'align': 'center', 'valign': 'vcenter', 'border': 1,
                'bg_color': '#BDD7EE', 'font_size': 9, 'text_wrap': True})
            # Wiersz 1: scalone naglowki blokow
            worksheet.write(sr, 0, '', fmt_blk_hdr)
            _cc = 1
            for _blk_lbl, _blk_cols in _bn_blks:
                _n = len(_blk_cols)
                worksheet.set_column(_cc, _cc + _n - 1, 12)
                if _n > 1:
                    worksheet.merge_range(sr, _cc, sr, _cc + _n - 1, _blk_lbl, fmt_blk_hdr)
                else:
                    worksheet.write(sr, _cc, _blk_lbl, fmt_blk_hdr)
                _cc += _n
            sr += 1
            # Wiersz 2: cat_display (bez prefiksu bloku)
            worksheet.write(sr, 0, 'Kategorie / Statystyki', fmt_cat_hdr)
            _cc = 1
            for _blk_lbl, _blk_cols in _bn_blks:
                for _col_name, _cat_disp in _blk_cols:
                    worksheet.write(sr, _cc, _cat_disp, fmt_cat_hdr)
                    _cc += 1
            sr += 1
        else:
            worksheet.write(sr, 0, "Kategorie / Statystyki", fmt_header)
            for c_idx, col_name in enumerate(df_export.columns):
                col_w = 16 if s_name not in ['Korelacje'] else 22
                worksheet.set_column(c_idx + 1, c_idx + 1, col_w)
                worksheet.write(sr, c_idx + 1, str(col_name), fmt_header)
            sr += 1

        # Data rows
        bold_rows = {'Suma', 'Braki danych', 'Braki danych (wykluczone z tabeli)',
                     'Og\u00f3\u0142em (Wa\u017cne)', 'Srednia', 'Odchylenie Std.', 'Baza (N)',
                     'Baza (N) / Suma (%)'}
        for r_idx, row_name in enumerate(df_export.index):
            # Banner: wiersz-naglowek pytania (cala linia pusta) -> scalony pasek
            if s_name == 'Banner' and df_export.iloc[r_idx].isna().all():
                if num_cols > 0:
                    worksheet.merge_range(sr + r_idx, 0, sr + r_idx, num_cols,
                                          str(row_name), fmt_qhdr)
                else:
                    worksheet.write(sr + r_idx, 0, str(row_name), fmt_qhdr)
                continue
            is_bold = str(row_name) in bold_rows
            idx_fmt = fmt_index_b if is_bold else fmt_index_n
            row_name_str = "" if (pd.isna(row_name) if not isinstance(row_name, str) else False) else str(row_name)
            worksheet.write(sr + r_idx, 0, row_name_str, idx_fmt)

            for c_idx, col_name in enumerate(df_export.columns):
                raw_val = df_export.iloc[r_idx, c_idx]
                col_str = str(col_name).lower()
                is_pct_col = "%" in col_str or "procent" in col_str

                try:
                    is_empty = pd.isna(raw_val)
                except:
                    is_empty = False
                if is_empty or str(raw_val).strip() in ('', 'nan', 'None'):
                    worksheet.write(sr + r_idx, c_idx + 1, "", fmt_empty)
                    continue

                if isinstance(raw_val, str):
                    worksheet.write(sr + r_idx, c_idx + 1, raw_val, fmt_str)
                elif s_name == 'Korelacje':
                    worksheet.write(sr + r_idx, c_idx + 1, str(raw_val), fmt_str)
                elif s_name == 'Opisowe':
                    worksheet.write(sr + r_idx, c_idx + 1, float(raw_val), fmt_float)
                elif s_name == '\u015arednie' and str(row_name) == 'Baza (N)':
                    worksheet.write(sr + r_idx, c_idx + 1, float(raw_val), fmt_n)
                elif s_name == '\u015arednie':
                    worksheet.write(sr + r_idx, c_idx + 1, float(raw_val), fmt_float)
                elif is_pct_col:
                    try:
                        worksheet.write(sr + r_idx, c_idx + 1, float(raw_val), fmt_pct)
                    except:
                        worksheet.write(sr + r_idx, c_idx + 1, str(raw_val), fmt_str)
                else:
                    try:
                        worksheet.write(sr + r_idx, c_idx + 1, float(raw_val), fmt_n)
                    except:
                        worksheet.write(sr + r_idx, c_idx + 1, str(raw_val), fmt_str)

        # \u2500\u2500 Native Excel chart (frequency tables only) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        if add_charts and s_name == 'Cz\u0119sto\u015bci':
            # Identify the % column index (1-based, col 0 = category labels)
            pct_col_idx = None
            for ci, col_name in enumerate(df_export.columns):
                cn_low = str(col_name).lower()
                if 'procent' in cn_low or '%' in cn_low:
                    pct_col_idx = ci + 1   # +1 because col 0 is the index label
                    break

            if pct_col_idx is not None:
                # Collect rows that should appear in the chart (exclude summary rows)
                chart_data_rows = []   # 0-based excel row numbers
                for r_idx, row_name in enumerate(df_export.index):
                    rn = str(row_name)
                    if rn in _chart_exclude:
                        continue
                    # Skip Box rows like [Top 2 Box]
                    if rn.startswith('[') and rn.endswith(']'):
                        continue
                    chart_data_rows.append(sr + r_idx)   # sr already advanced past headers

                if len(chart_data_rows) >= 2:
                    first_row = chart_data_rows[0]
                    last_row  = chart_data_rows[-1]

                    chart = workbook.add_chart({'type': 'bar'})   # horizontal bars
                    chart.add_series({
                        'name':       chart_title[:60],
                        'categories': [sheet_name, first_row, 0,          last_row, 0],
                        'values':     [sheet_name, first_row, pct_col_idx, last_row, pct_col_idx],
                        'fill':       {'color': '#2E75B6'},
                        'border':     {'color': '#1F4E79'},
                        'gap':        60,
                        'data_labels': {
                            'value':      True,
                            'num_format': '0.0"%"',
                            'font':       {'size': 9},
                            'position':   'outside_end',
                        },
                    })
                    chart.set_title({
                        'name':    chart_title[:80],
                        'overlay': False,
                    })
                    # X axis: no title, no tick labels, no gridlines
                    chart.set_x_axis({
                        'name':         '',
                        'min':           0,
                        'num_font':     {'size': 1, 'color': '#FFFFFF'},  # invisible labels
                        'major_gridlines': {'visible': False},
                        'minor_gridlines': {'visible': False},
                        'major_tick_mark': 'none',
                        'minor_tick_mark': 'none',
                        'line':          {'none': True},
                    })
                    # Y axis: categories top-to-bottom, no gridlines
                    chart.set_y_axis({
                        'reverse':         True,
                        'num_font':        {'size': 9},
                        'major_gridlines': {'visible': False},
                        'minor_gridlines': {'visible': False},
                        'major_tick_mark': 'none',
                        'minor_tick_mark': 'none',
                        'line':            {'none': True},
                    })
                    chart.set_legend({'none': True})
                    chart.set_plotarea({'border': {'none': True}})
                    chart.set_chartarea({'border': {'color': '#D6E4F0'}})

                    # Height: match the table exactly.
                    # Excel default row height = 15pt = 20px.
                    # Table occupies: 1 title row + 1 header row + len(df_export) data rows + 1 blank = len+3 rows
                    # We use 20px per row as a close approximation.
                    table_rows   = len(df_export) + 2   # title + header + data rows
                    c_height     = max(180, table_rows * 20)
                    chart.set_size({'width': 480, 'height': c_height})

                    # Insert aligned with table title row, to the right
                    insert_col = num_cols + 2
                    title_row  = title_row_map[title]
                    worksheet.insert_chart(title_row, insert_col, chart,
                                           {'x_offset': 5, 'y_offset': 0})

        # \u2500\u2500 Advance row pointer \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        sr += len(df_export) + 1

        # Chi-square note
        if s_name == 'Krzy\u017cowe' and title in st.session_state.chi_results:
            worksheet.write(sr, 0, st.session_state.chi_results[title],
                            workbook.add_format({'italic': True, 'font_color': '#595959'}))
            sr += 1
        sr += 2  # blank rows between tables

    return title_row_map   # {title: starting excel row} for ToC hyperlinks


def export_regression_to_excel(writer, regression_results, var_labels):
    workbook = writer.book
    worksheet = workbook.add_worksheet('Regresja')

    fmt_title   = workbook.add_format({'bold': True, 'bg_color': '#1F4E79', 'font_color': 'white', 'align': 'center', 'valign': 'vcenter', 'border': 1, 'font_size': 11})
    fmt_section = workbook.add_format({'bold': True, 'bg_color': '#D6E4F0', 'border': 1, 'align': 'left'})
    fmt_header  = workbook.add_format({'bold': True, 'bg_color': '#F2F2F2', 'border': 1, 'align': 'center', 'text_wrap': True})
    fmt_label   = workbook.add_format({'bold': True, 'border': 1})
    fmt_val     = workbook.add_format({'border': 1, 'align': 'right', 'num_format': '#,##0.000'})
    fmt_int     = workbook.add_format({'border': 1, 'align': 'right', 'num_format': '#,##0'})
    fmt_warn    = workbook.add_format({'border': 1, 'align': 'right', 'num_format': '#,##0.000', 'font_color': '#C00000'})
    fmt_ok      = workbook.add_format({'border': 1, 'align': 'right', 'num_format': '#,##0.000', 'font_color': '#006100'})
    fmt_dash    = workbook.add_format({'border': 1, 'align': 'center'})

    worksheet.set_column(0, 0, 38)
    for i in range(1, 10): worksheet.set_column(i, i, 16)

    row = 0
    for res in regression_results:
        if 'error' in res:
            worksheet.write(row, 0, f"B\u0141\u0104D: {res['error']}", fmt_section)
            row += 2
            continue
        dep_label = var_labels.get(res['dep_var'], res['dep_var'])
        _grp_s = f"  \u2014  \U0001f500 {res['group_label']}" if res.get('group_label') else ""
        worksheet.merge_range(row, 0, row, 8, f"REGRESJA OLS -- Zmienna zale\u017cna: [{res['dep_var']}] {dep_label}  |  Blok {res['Blok']}{_grp_s}", fmt_title)
        row += 1
        worksheet.write(row, 0, "Podsumowanie Modelu", fmt_section)
        row += 1
        for field, val, fmt in [
            ('N (obserwacje)', res['N'], fmt_int),
            ('R', res['R'], fmt_val),
            ('R2', res['R2'], fmt_val),
            ('Skor_R2', res['Skor_R2'], fmt_val),
            ('\u0394R\u00b2 (zmiana R\u00b2)', res['Delta_R2'], fmt_val),
            ('F modelu', res['F modelu'], fmt_val),
            ('p (F modelu)', res['p (F modelu)'], fmt_val),
            ('F zmiany', res['F zmiany'], fmt_val),
            ('p (F zmiany)', res['p (F zmiany)'], fmt_val),
        ]:
            worksheet.write(row, 0, field, fmt_label)
            try:
                v = float(val)
                worksheet.write(row, 1, v, fmt)
            except:
                worksheet.write(row, 1, '--', fmt_dash)
            row += 1
        row += 1
        worksheet.write(row, 0, "Wsp\u00f3\u0142czynniki Regresji", fmt_section)
        row += 1
        for ci, h in enumerate(['Zmienna', 'B', 'B\u0142\u0105d std. B', 'Beta (std.)', 't', 'p-value', 'VIF', 'Tolerancja']):
            worksheet.write(row, ci, h, fmt_header)
        row += 1
        for _, r_data in res['coef_df'].iterrows():
            vn = r_data['Zmienna']
            worksheet.write(row, 0, f"[{vn}] {var_labels.get(vn, vn)}", fmt_label)
            worksheet.write(row, 1, float(r_data['B']), fmt_val)
            worksheet.write(row, 2, float(r_data['B\u0142\u0105d std. B']), fmt_val)
            try: worksheet.write(row, 3, float(r_data['Beta (std.)']), fmt_val)
            except: worksheet.write(row, 3, '--', fmt_dash)
            worksheet.write(row, 4, float(r_data['t']), fmt_val)
            p = r_data['p-value']
            try:
                pf = float(p)
                worksheet.write(row, 5, pf, fmt_ok if pf < 0.05 else fmt_val)
            except: worksheet.write(row, 5, '--', fmt_dash)
            vif = r_data['VIF']
            try:
                vf = float(vif)
                vif_fmt = fmt_warn if vf > 10 else fmt_val
                worksheet.write(row, 6, vf, vif_fmt)
                worksheet.write(row, 7, float(r_data['Tolerancja']), vif_fmt)
            except:
                worksheet.write(row, 6, '--', fmt_dash)
                worksheet.write(row, 7, '--', fmt_dash)
            row += 1
        worksheet.write(row, 0, "VIF > 10 = problem ze wsp\u00f3\u0142liniowo\u015bci\u0105  |  p < 0.05 = istotne statystycznie",
                        workbook.add_format({'italic': True, 'font_color': '#595959'}))
        row += 3


def export_anova_to_excel(writer, anova_results, var_labels):
    workbook = writer.book
    worksheet = workbook.add_worksheet('ANOVA')
    fmt_title   = workbook.add_format({'bold': True, 'bg_color': '#1F4E79', 'font_color': 'white', 'border': 1, 'align': 'center', 'font_size': 11})
    fmt_section = workbook.add_format({'bold': True, 'bg_color': '#D6E4F0', 'border': 1})
    fmt_header  = workbook.add_format({'bold': True, 'bg_color': '#F2F2F2', 'border': 1, 'align': 'center'})
    fmt_label   = workbook.add_format({'bold': True, 'border': 1})
    fmt_val     = workbook.add_format({'border': 1, 'align': 'right', 'num_format': '#,##0.000'})
    fmt_int     = workbook.add_format({'border': 1, 'align': 'right', 'num_format': '#,##0'})
    fmt_str     = workbook.add_format({'border': 1, 'align': 'center'})
    worksheet.set_column(0, 0, 30)
    for i in range(1, 8): worksheet.set_column(i, i, 16)
    row = 0
    for res in anova_results:
        dep_l = var_labels.get(res['dep_var'], res['dep_var'])
        grp_l = var_labels.get(res['group_var'], res['group_var'])
        _grp_s = f"  \u2014  \U0001f500 {res['group_label']}" if res.get('group_label') else ""
        worksheet.merge_range(row, 0, row, 6, f"ANOVA -- Zmienna zale\u017cna: {dep_l}  |  Czynnik: {grp_l}{_grp_s}", fmt_title)
        row += 2
        worksheet.write(row, 0, "Tabela ANOVA", fmt_section)
        row += 1
        for ci, h in enumerate(['\u0179r\u00f3d\u0142o', 'SS', 'df', 'MS', 'F', 'p-value', 'Eta\u00b2']):
            worksheet.write(row, ci, h, fmt_header)
        row += 1
        worksheet.write(row, 0, "Mi\u0119dzy grupami", fmt_label)
        worksheet.write(row, 1, res['ss_between'], fmt_val)
        worksheet.write(row, 2, res['df_between'], fmt_int)
        worksheet.write(row, 3, res['ms_between'], fmt_val)
        worksheet.write(row, 4, res['F'], fmt_val)
        worksheet.write(row, 5, res['p'], fmt_val)
        worksheet.write(row, 6, res['eta2'], fmt_val)
        row += 1
        worksheet.write(row, 0, "Wewn\u0105trz grup", fmt_label)
        worksheet.write(row, 1, res['ss_within'], fmt_val)
        worksheet.write(row, 2, res['df_within'], fmt_int)
        worksheet.write(row, 3, res['ms_within'], fmt_val)
        row += 2
        # Descriptives
        worksheet.write(row, 0, "Statystyki opisowe wg grupy", fmt_section)
        row += 1
        for ci, h in enumerate(res['desc_df'].columns):
            worksheet.write(row, ci, h, fmt_header)
        row += 1
        for _, r_d in res['desc_df'].iterrows():
            for ci, v in enumerate(r_d):
                try: worksheet.write(row, ci, float(v), fmt_val)
                except: worksheet.write(row, ci, str(v), fmt_str)
            row += 1
        row += 1
        # Post-hoc
        if not res['posthoc_df'].empty:
            worksheet.write(row, 0, "Test post-hoc: Tukey HSD", fmt_section)
            row += 1
            for ci, h in enumerate(res['posthoc_df'].columns):
                worksheet.write(row, ci, h, fmt_header)
            row += 1
            for _, r_d in res['posthoc_df'].iterrows():
                for ci, v in enumerate(r_d):
                    try: worksheet.write(row, ci, float(v), fmt_val)
                    except: worksheet.write(row, ci, str(v), fmt_str)
                row += 1
        row += 3


def export_factor_to_excel(writer, factor_results, var_labels):
    workbook = writer.book
    worksheet = workbook.add_worksheet('Anal. Czynnikowa')
    fmt_title   = workbook.add_format({'bold': True, 'bg_color': '#1F4E79', 'font_color': 'white', 'border': 1, 'align': 'center', 'font_size': 11})
    fmt_section = workbook.add_format({'bold': True, 'bg_color': '#D6E4F0', 'border': 1})
    fmt_header  = workbook.add_format({'bold': True, 'bg_color': '#F2F2F2', 'border': 1, 'align': 'center'})
    fmt_label   = workbook.add_format({'bold': True, 'border': 1})
    fmt_val     = workbook.add_format({'border': 1, 'align': 'right', 'num_format': '#,##0.000'})
    fmt_hi      = workbook.add_format({'border': 1, 'align': 'right', 'num_format': '#,##0.000', 'bold': True, 'bg_color': '#E2EFDA'})
    worksheet.set_column(0, 0, 38)
    for i in range(1, 12): worksheet.set_column(i, i, 14)
    row = 0
    for res in factor_results:
        _grp_s = f"  \u2014  \U0001f500 {res['group_label']}" if res.get('group_label') else ""
        worksheet.merge_range(row, 0, row, res['loadings'].shape[1], f"ANALIZA CZYNNIKOWA -- Rotacja: {res['rotation'].upper()}  |  N={res['n']}{_grp_s}", fmt_title)
        row += 1
        # KMO and Bartlett
        worksheet.write(row, 0, "Adekwatno\u015b\u0107 pr\u00f3by KMO", fmt_label)
        worksheet.write(row, 1, res['kmo'], fmt_val)
        row += 1
        worksheet.write(row, 0, "Test sferyczno\u015bci Bartletta (Chi\u00b2)", fmt_label)
        worksheet.write(row, 1, res['bartlett_chi2'], fmt_val)
        row += 1
        worksheet.write(row, 0, "Test sferyczno\u015bci Bartletta (p)", fmt_label)
        worksheet.write(row, 1, res['bartlett_p'], fmt_val)
        row += 2
        # Loadings
        worksheet.write(row, 0, "Macierz \u0141adunk\u00f3w Czynnikowych", fmt_section)
        row += 1
        worksheet.write(row, 0, "Zmienna", fmt_header)
        for ci, col in enumerate(res['loadings'].columns):
            worksheet.write(row, ci + 1, col, fmt_header)
        worksheet.write(row, len(res['loadings'].columns) + 1, "Komunalno\u015b\u0107 (h\u00b2)", fmt_header)
        row += 1
        for var in res['loadings'].index:
            worksheet.write(row, 0, f"[{var}] {var_labels.get(var, var)}", fmt_label)
            for ci, col in enumerate(res['loadings'].columns):
                val = res['loadings'].loc[var, col]
                fmt_use = fmt_hi if abs(val) >= 0.4 else fmt_val
                worksheet.write(row, ci + 1, float(val), fmt_use)
            worksheet.write(row, len(res['loadings'].columns) + 1,
                            float(res['communalities'].loc[var, 'Komunalnosc (h2)']), fmt_val)
            row += 1
        row += 2
        # Variance explained
        worksheet.write(row, 0, "Wyja\u015bniona Wariancja", fmt_section)
        row += 1
        for ci, col in enumerate(['', 'SS \u0141adunk\u00f3w', '% wariancji', 'Skumulowany %']):
            worksheet.write(row, ci, col, fmt_header)
        row += 1
        for idx, r_d in res['variance'].iterrows():
            worksheet.write(row, 0, str(idx), fmt_label)
            for ci, v in enumerate(r_d):
                worksheet.write(row, ci + 1, float(v), fmt_val)
            row += 1
        row += 3


def export_matrix_to_excel(writer, matrix_results, var_labels):
    """
    Export matrix/battery frequency tables.
    Layout: Rows = scale values, Columns = subquestions (N | %).
    """
    workbook  = writer.book
    worksheet = workbook.add_worksheet('Pytania Matrycowe')

    fmt_title    = workbook.add_format({'bold': True, 'bg_color': '#1F4E79', 'font_color': 'white',
                                         'align': 'center', 'valign': 'vcenter', 'border': 1, 'font_size': 11})
    fmt_sub_hdr  = workbook.add_format({'bold': True, 'bg_color': '#2E75B6', 'font_color': 'white',
                                         'align': 'center', 'border': 1, 'text_wrap': True})
    fmt_np_hdr   = workbook.add_format({'bold': True, 'bg_color': '#D6E4F0',
                                         'align': 'center', 'border': 1})
    fmt_val_lbl  = workbook.add_format({'bold': True, 'border': 1, 'align': 'left', 'bg_color': '#F2F2F2'})
    fmt_base_lbl = workbook.add_format({'bold': True, 'border': 1, 'align': 'left', 'bg_color': '#E2EFDA'})
    fmt_suma_lbl = workbook.add_format({'bold': True, 'border': 1, 'align': 'left', 'bg_color': '#D6E4F0', 'italic': True})
    fmt_n        = workbook.add_format({'num_format': '#,##0',  'border': 1, 'align': 'right'})
    fmt_pct      = workbook.add_format({'num_format': '0.0"%"', 'border': 1, 'align': 'right'})
    fmt_base_n   = workbook.add_format({'num_format': '#,##0',  'border': 1, 'align': 'right',
                                         'bold': True, 'bg_color': '#E2EFDA'})
    fmt_suma_pct = workbook.add_format({'num_format': '0.0"%"', 'border': 1, 'align': 'right',
                                         'bold': True, 'bg_color': '#D6E4F0', 'italic': True})
    fmt_empty    = workbook.add_format({'border': 1})

    worksheet.set_column(0, 0, 14)   # row-index column (scale values)

    row = 0
    for entry in matrix_results:
        name         = entry['name']
        df_matrix    = entry['df']
        all_cats     = entry['cats']
        sub_labels   = entry['sub_labels']
        display_mode = entry.get('display_mode', 'N + %')   # default: show both

        n_subs = len(sub_labels)
        # Determine how many columns per subquestion based on display mode
        cols_per_sub = 1 if display_mode in ('Tylko N', 'Tylko %') else 2
        total_data_cols = n_subs * cols_per_sub
        total_cols = total_data_cols

        # -- Title --
        if total_cols > 0:
            worksheet.merge_range(row, 0, row, total_cols, f"Pytanie matrycowe: {name}", fmt_title)
        else:
            worksheet.write(row, 0, f"Pytanie matrycowe: {name}", fmt_title)
        row += 1

        # -- Row 1: subquestion labels --
        worksheet.write(row, 0, "Warto\u015b\u0107 \\ Subpytanie", fmt_sub_hdr)
        col_cur = 1
        for sub_lbl in sub_labels:
            disp = sub_lbl
            if display_mode == 'N + %':
                worksheet.merge_range(row, col_cur, row, col_cur + 1, disp, fmt_sub_hdr)
                worksheet.set_column(col_cur,     col_cur,     11)
                worksheet.set_column(col_cur + 1, col_cur + 1, 9)
                col_cur += 2
            else:
                worksheet.write(row, col_cur, disp, fmt_sub_hdr)
                worksheet.set_column(col_cur, col_cur, 12)
                col_cur += 1
        row += 1

        # -- Row 2: N / % sub-headers --
        worksheet.write(row, 0, "", fmt_np_hdr)
        col_cur = 1
        for _ in sub_labels:
            if display_mode == 'N + %':
                worksheet.write(row, col_cur,     "N",  fmt_np_hdr)
                worksheet.write(row, col_cur + 1, "%",  fmt_np_hdr)
                col_cur += 2
            elif display_mode == 'Tylko N':
                worksheet.write(row, col_cur, "N", fmt_np_hdr)
                col_cur += 1
            else:
                worksheet.write(row, col_cur, "%", fmt_np_hdr)
                col_cur += 1
        row += 1

        # -- Data rows --
        for cat_val in all_cats:
            worksheet.write(row, 0, str(cat_val), fmt_val_lbl)
            col_cur = 1
            for sub_lbl in sub_labels:
                if display_mode == 'N + %':
                    n_val   = df_matrix.loc[cat_val, f"{sub_lbl} [N]"]
                    pct_val = df_matrix.loc[cat_val, f"{sub_lbl} [%]"]
                    try: worksheet.write(row, col_cur,     float(n_val),   fmt_n)
                    except: worksheet.write(row, col_cur,     "", fmt_empty)
                    try: worksheet.write(row, col_cur + 1, float(pct_val), fmt_pct)
                    except: worksheet.write(row, col_cur + 1, "", fmt_empty)
                    col_cur += 2
                elif display_mode == 'Tylko N':
                    n_val = df_matrix.loc[cat_val, f"{sub_lbl} [N]"]
                    try: worksheet.write(row, col_cur, float(n_val), fmt_n)
                    except: worksheet.write(row, col_cur, "", fmt_empty)
                    col_cur += 1
                else:  # Tylko %
                    pct_val = df_matrix.loc[cat_val, f"{sub_lbl} [%]"]
                    try: worksheet.write(row, col_cur, float(pct_val), fmt_pct)
                    except: worksheet.write(row, col_cur, "", fmt_empty)
                    col_cur += 1
            row += 1

        # -- Single combined summary row: "Baza (N) / Suma (%)" --
        # N and % sit side by side, matching the frequency table style
        SUMROW = "Baza (N) / Suma (%)"
        worksheet.write(row, 0, SUMROW, fmt_suma_lbl)
        col_cur = 1
        for sub_lbl in sub_labels:
            base_val = df_matrix.loc[SUMROW, f"{sub_lbl} [N]"]
            suma_val = df_matrix.loc[SUMROW, f"{sub_lbl} [%]"]
            if display_mode == 'N + %':
                try: worksheet.write(row, col_cur,     float(base_val), fmt_base_n)
                except: worksheet.write(row, col_cur,     "", fmt_empty)
                try: worksheet.write(row, col_cur + 1, float(suma_val), fmt_suma_pct)
                except: worksheet.write(row, col_cur + 1, "", fmt_empty)
                col_cur += 2
            elif display_mode == 'Tylko N':
                try: worksheet.write(row, col_cur, float(base_val), fmt_base_n)
                except: worksheet.write(row, col_cur, "", fmt_empty)
                col_cur += 1
            else:  # Tylko %
                try: worksheet.write(row, col_cur, float(suma_val), fmt_suma_pct)
                except: worksheet.write(row, col_cur, "", fmt_empty)
                col_cur += 1
        row += 3   # gap between batteries


def write_db_sheet(writer, sheet_label, data_df, var_labels, hdr_color='#1F4E79', header_mode='names'):
    """Write a single database sheet into an already-open ExcelWriter.
    Row 0 = column names (header), data from row 1 onwards.
    header_mode: 'names' (original col names) or 'labels' (var_labels)"""
    workbook = writer.book
    ws = workbook.add_worksheet(sheet_label[:31])
    fmt_h = workbook.add_format({
        'bold': True, 'bg_color': hdr_color, 'font_color': 'white',
        'border': 1, 'align': 'center',
    })
    for ci, col in enumerate(data_df.columns):
        if header_mode == 'labels':
            header_text = var_labels.get(col, col) or col
        else:
            header_text = col
        ws.write(0, ci, header_text, fmt_h)
        ws.set_column(ci, ci, 16)
    for ri, (_, row_data) in enumerate(data_df.iterrows()):
        for ci, val in enumerate(row_data):
            try:
                is_na = pd.isna(val)
            except Exception:
                is_na = False
            if is_na:
                ws.write(ri + 1, ci, '')
            elif isinstance(val, (int, float, np.integer, np.floating)):
                ws.write(ri + 1, ci, float(val))
            else:
                s = str(val)
                ws.write(ri + 1, ci, '' if s in ('nan', 'None', '<NA>') else s)


def export_db_to_excel(df_raw, df_labeled, var_labels, header_mode='names'):
    """Standalone download: both sheets in one file.
    header_mode: 'names' or 'labels'"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        write_db_sheet(writer, 'Baza z etykietami',        df_labeled, var_labels, '#1F4E79', header_mode)
        write_db_sheet(writer, 'Baza surowa (numeryczna)', df_raw,     var_labels, '#2E75B6', header_mode)
    return output.getvalue()

# -------------------------------------------------------------
# =============================================================
# CONJOINT ANALYSIS
# =============================================================

def run_conjoint_rating(df_raw, rating_var, attribute_vars):
    """Rating-based Conjoint via OLS. Returns part-worth utilities and importance."""
    from sklearn.preprocessing import LabelEncoder
    df_c = df_raw[[rating_var] + attribute_vars].dropna()
    if len(df_c) < 10:
        return None, "Za ma\u0142o obserwacji."
    y = df_c[rating_var].astype(float)
    # Dummy-encode categorical attributes; numeric attributes treated as linear
    X_parts = []
    attr_info = {}
    for attr in attribute_vars:
        col = df_c[attr]
        if col.dtype == object or col.nunique() <= 8:
            dummies = pd.get_dummies(col.astype(str), prefix=attr, drop_first=False)
            X_parts.append(dummies)
            attr_info[attr] = {'type': 'categorical', 'levels': list(dummies.columns)}
        else:
            X_parts.append(col.rename(attr).to_frame())
            attr_info[attr] = {'type': 'numeric', 'levels': [attr]}
    X = pd.concat(X_parts, axis=1).astype(float)
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()
    # Part-worth utilities
    utilities = {}
    for attr, info in attr_info.items():
        lvl_utils = {}
        for lv in info['levels']:
            lvl_utils[lv] = model.params.get(lv, 0.0)
        # Zero-center (sum-to-zero coding adjustment)
        mean_u = np.mean(list(lvl_utils.values()))
        lvl_utils = {k: v - mean_u for k, v in lvl_utils.items()}
        utilities[attr] = lvl_utils
    # Relative importance: range of utilities per attribute
    ranges = {attr: max(u.values()) - min(u.values()) for attr, u in utilities.items()}
    total_range = sum(ranges.values())
    importance = {attr: (r / total_range * 100) if total_range > 0 else 0
                  for attr, r in ranges.items()}
    return {
        'method': 'Rating-based (OLS)',
        'rating_var': rating_var,
        'attribute_vars': attribute_vars,
        'n': len(df_c),
        'r2': model.rsquared,
        'r2_adj': model.rsquared_adj,
        'f': model.fvalue,
        'p': model.f_pvalue,
        'utilities': utilities,
        'importance': importance,
        'model': model,
        'attr_info': attr_info,
    }, None


def run_conjoint_cbc(df_raw, choice_var, attribute_vars):
    """Choice-Based Conjoint via logistic regression. choice_var = 0/1."""
    from sklearn.linear_model import LogisticRegression
    df_c = df_raw[[choice_var] + attribute_vars].dropna()
    if len(df_c) < 20:
        return None, "Za ma\u0142o obserwacji (min. 20)."
    y = df_c[choice_var].astype(int)
    X_parts = []
    attr_info = {}
    for attr in attribute_vars:
        col = df_c[attr]
        if col.dtype == object or col.nunique() <= 8:
            dummies = pd.get_dummies(col.astype(str), prefix=attr, drop_first=True)
            X_parts.append(dummies)
            attr_info[attr] = {'type': 'categorical', 'levels': list(dummies.columns)}
        else:
            X_parts.append(col.rename(attr).to_frame())
            attr_info[attr] = {'type': 'numeric', 'levels': [attr]}
    X = pd.concat(X_parts, axis=1).astype(float)
    X_const = sm.add_constant(X)
    try:
        model = sm.Logit(y, X_const).fit(disp=False)
    except Exception as e:
        return None, str(e)
    utilities = {}
    for attr, info in attr_info.items():
        lvl_utils = {}
        for lv in info['levels']:
            lvl_utils[lv] = model.params.get(lv, 0.0)
        mean_u = np.mean(list(lvl_utils.values())) if lvl_utils else 0
        utilities[attr] = {k: v - mean_u for k, v in lvl_utils.items()}
    ranges = {attr: max(u.values()) - min(u.values()) if u else 0 for attr, u in utilities.items()}
    total_range = sum(ranges.values())
    importance = {attr: (r / total_range * 100) if total_range > 0 else 0
                  for attr, r in ranges.items()}
    return {
        'method': 'CBC (Logit)',
        'choice_var': choice_var,
        'attribute_vars': attribute_vars,
        'n': len(df_c),
        'llr': model.llr,
        'llr_pvalue': model.llr_pvalue,
        'pseudo_r2': model.prsquared,
        'utilities': utilities,
        'importance': importance,
        'model': model,
        'attr_info': attr_info,
    }, None


def export_conjoint_to_excel(writer, conjoint_results, var_labels, meta_vvl=None, custom_val_labels=None):
    workbook = writer.book
    ws = workbook.add_worksheet('Conjoint')
    fmt_t  = workbook.add_format({'bold': True, 'bg_color': '#1F4E79', 'font_color': 'white',
                                   'border': 1, 'align': 'center', 'font_size': 11})
    fmt_s  = workbook.add_format({'bold': True, 'bg_color': '#D6E4F0', 'border': 1})
    fmt_h  = workbook.add_format({'bold': True, 'bg_color': '#F2F2F2', 'border': 1, 'align': 'center'})
    fmt_lbl= workbook.add_format({'border': 1, 'bold': True})
    fmt_val= workbook.add_format({'border': 1, 'align': 'right', 'num_format': '#,##0.000'})
    fmt_pct= workbook.add_format({'border': 1, 'align': 'right', 'num_format': '0.0"%"'})
    fmt_str= workbook.add_format({'border': 1})
    ws.set_column(0, 0, 35)
    ws.set_column(1, 5, 16)

    def _nice_level(attr, raw_lvl):
        """Convert dummy-column name like 'Q1_A1' -> 'Q1_zdecydowanie'."""
        s = str(raw_lvl)
        prefix = f"{attr}_"
        code_part = s[len(prefix):] if s.startswith(prefix) else s
        _cvl = (custom_val_labels or {}).get(attr, {})
        _vvl = (meta_vvl or {}).get(attr, {})
        lbl_txt = _cvl.get(code_part, _cvl.get(str(code_part), ""))
        if not lbl_txt:
            for key_variant in (code_part, str(code_part)):
                try:
                    fv = float(key_variant)
                    lbl_txt = _vvl.get(fv, _vvl.get(int(fv), ""))
                    if lbl_txt:
                        break
                except (ValueError, TypeError):
                    pass
            if not lbl_txt:
                lbl_txt = _vvl.get(code_part, "")
        return f"{attr}_{lbl_txt}" if lbl_txt else s

    row = 0
    for res in conjoint_results:
        if res.get('error'):
            ws.write(row, 0, f"B\u0141\u0104D: {res['error']}", fmt_s); row += 2; continue
        _grp_s = f"  \u2014  \U0001f500 {res['group_label']}" if res.get('group_label') else ""
        ws.merge_range(row, 0, row, 4, f"CONJOINT -- {res['method']}{_grp_s}", fmt_t); row += 1
        # Model summary
        ws.write(row, 0, "N", fmt_lbl)
        ws.write(row, 1, res['n'], workbook.add_format({'border':1,'align':'right','num_format':'#,##0'}))
        row += 1
        if 'r2' in res:
            ws.write(row, 0, "R\u00b2", fmt_lbl); ws.write(row, 1, res['r2'], fmt_val); row += 1
            ws.write(row, 0, "R\u00b2 skor.", fmt_lbl); ws.write(row, 1, res['r2_adj'], fmt_val); row += 1
            ws.write(row, 0, "F / p", fmt_lbl)
            ws.write(row, 1, res['f'], fmt_val); ws.write(row, 2, res['p'], fmt_val); row += 1
        if 'pseudo_r2' in res:
            ws.write(row, 0, "Pseudo R\u00b2 (McFadden)", fmt_lbl); ws.write(row, 1, res['pseudo_r2'], fmt_val); row += 1
        row += 1
        # Importance
        ws.write(row, 0, "Wa\u017cno\u015b\u0107 atrybut\u00f3w (%)", fmt_s); row += 1
        for attr, imp in sorted(res['importance'].items(), key=lambda x: -x[1]):
            ws.write(row, 0, f"[{attr}] {var_labels.get(attr, attr)}", fmt_lbl)
            ws.write(row, 1, imp, fmt_pct); row += 1
        row += 1
        # Utilities
        ws.write(row, 0, "U\u017cyteczno\u015bci cz\u0105stkowe (part-worth utilities)", fmt_s); row += 1
        ws.write(row, 0, "Atrybut / Poziom", fmt_h)
        ws.write(row, 1, "U\u017cyteczno\u015b\u0107", fmt_h); row += 1
        for attr, utils in res['utilities'].items():
            ws.write(row, 0, f"[{attr}] {var_labels.get(attr, attr)}", fmt_lbl)
            ws.write(row, 1, "", fmt_str); row += 1
            for level, util in sorted(utils.items(), key=lambda x: -x[1]):
                ws.write(row, 0, f"  {_nice_level(attr, level)}", fmt_str)
                ws.write(row, 1, util, fmt_val); row += 1
        row += 3


# =============================================================
# MAXDIFF ANALYSIS
# =============================================================

def run_maxdiff(df_raw, task_pairs, item_values):
    """
    MaxDiff scoring from paired Best/Worst columns.
    task_pairs: list of (best_col, worst_col) tuples
    item_values: list of unique item labels (strings) that appear in those columns
    Returns: DataFrame with item scores and ranks.
    """
    n_resp = len(df_raw)
    counts = {item: {'best': 0, 'worst': 0, 'shown': 0} for item in item_values}
    for best_col, worst_col in task_pairs:
        if best_col not in df_raw.columns or worst_col not in df_raw.columns:
            continue
        best_series  = df_raw[best_col].dropna().astype(str)
        worst_series = df_raw[worst_col].dropna().astype(str)
        for item in item_values:
            counts[item]['best']  += (best_series  == str(item)).sum()
            counts[item]['worst'] += (worst_series == str(item)).sum()
            counts[item]['shown'] += ((best_series == str(item)) | (worst_series == str(item))).sum()
    rows = []
    for item in item_values:
        b = counts[item]['best']
        w = counts[item]['worst']
        shown = counts[item]['shown']
        bw_score = b - w
        bw_pct   = bw_score / n_resp * 100 if n_resp > 0 else 0
        rows.append({'Item': item, 'Best [N]': b, 'Worst [N]': w,
                     'B-W Score': bw_score, 'B-W Score (%)': round(bw_pct, 2),
                     'Pokazano [N]': shown})
    df_scores = pd.DataFrame(rows).sort_values('B-W Score', ascending=False).reset_index(drop=True)
    df_scores.insert(0, 'Ranking', range(1, len(df_scores) + 1))
    # Rescale to 0-100 (most positive = 100)
    mn, mx = df_scores['B-W Score'].min(), df_scores['B-W Score'].max()
    if mx > mn:
        df_scores['Wynik standaryzowany (0-100)'] = ((df_scores['B-W Score'] - mn) / (mx - mn) * 100).round(1)
    else:
        df_scores['Wynik standaryzowany (0-100)'] = 50.0
    return df_scores


def export_maxdiff_to_excel(writer, maxdiff_results, var_labels):
    workbook = writer.book
    ws = workbook.add_worksheet('MaxDiff')
    fmt_t   = workbook.add_format({'bold': True, 'bg_color': '#1F4E79', 'font_color': 'white',
                                    'border': 1, 'align': 'center', 'font_size': 11})
    fmt_h   = workbook.add_format({'bold': True, 'bg_color': '#F2F2F2', 'border': 1, 'align': 'center'})
    fmt_lbl = workbook.add_format({'border': 1})
    fmt_n   = workbook.add_format({'border': 1, 'align': 'right', 'num_format': '#,##0'})
    fmt_val = workbook.add_format({'border': 1, 'align': 'right', 'num_format': '#,##0.00'})
    fmt_pct = workbook.add_format({'border': 1, 'align': 'right', 'num_format': '0.0'})
    ws.set_column(0, 0, 8); ws.set_column(1, 1, 35)
    ws.set_column(2, 7, 18)
    row = 0
    for res in maxdiff_results:
        ws.merge_range(row, 0, row, 6, f"MaxDiff -- {res['name']}", fmt_t); row += 1
        ws.write(row, 0, f"N respondent\u00f3w: {res['n_resp']}  |  Liczba zestaw\u00f3w: {res['n_tasks']}",
                 workbook.add_format({'italic': True})); row += 2
        cols_out = ['Ranking', 'Item', 'Best [N]', 'Worst [N]', 'B-W Score', 'B-W Score (%)', 'Wynik standaryzowany (0-100)']
        for ci, col in enumerate(cols_out):
            ws.write(row, ci, col, fmt_h)
        row += 1
        df_s = res['scores']
        for _, r in df_s.iterrows():
            ws.write(row, 0, int(r['Ranking']), fmt_n)
            ws.write(row, 1, str(r['Item']), fmt_lbl)
            ws.write(row, 2, int(r['Best [N]']), fmt_n)
            ws.write(row, 3, int(r['Worst [N]']), fmt_n)
            ws.write(row, 4, float(r['B-W Score']), fmt_val)
            ws.write(row, 5, float(r['B-W Score (%)']), fmt_pct)
            ws.write(row, 6, float(r['Wynik standaryzowany (0-100)']), fmt_pct)
            row += 1
        row += 3


# SESSION STATE
# -------------------------------------------------------------
if 'authenticated'          not in st.session_state: st.session_state.authenticated = False
if 'current_user_id'        not in st.session_state: st.session_state.current_user_id = None
if 'current_user_name'      not in st.session_state: st.session_state.current_user_name = ""
if 'current_user_role'      not in st.session_state: st.session_state.current_user_role = ""
if 'current_user_perms'     not in st.session_state: st.session_state.current_user_perms = {}
if 'session_token'          not in st.session_state: st.session_state.session_token = None
if 'session_db_id'          not in st.session_state: st.session_state.session_db_id = None
if 'last_activity_ts'       not in st.session_state: st.session_state.last_activity_ts = time.time()
if 'must_change_password'   not in st.session_state: st.session_state.must_change_password = False
if 'current_user_ip'        not in st.session_state: st.session_state.current_user_ip = ""
if 'mrs_sets'            not in st.session_state: st.session_state.mrs_sets = {}
if 'matrix_sets'         not in st.session_state: st.session_state.matrix_sets = {}
if 'matrix_results'      not in st.session_state: st.session_state.matrix_results = []
if 'custom_var_labels'   not in st.session_state: st.session_state.custom_var_labels = {}
if 'custom_val_labels'   not in st.session_state: st.session_state.custom_val_labels = {}
if 'value_orders'        not in st.session_state: st.session_state.value_orders = {}
if 'user_cleared_val_labels' not in st.session_state: st.session_state.user_cleared_val_labels = set()
if 'ppt_chart_templates'  not in st.session_state: st.session_state.ppt_chart_templates = {}
if 'box_sets'            not in st.session_state: st.session_state.box_sets = defaultdict(dict)
if 'segmentations'       not in st.session_state: st.session_state.segmentations = []
if 'hclust_results'      not in st.session_state: st.session_state.hclust_results = []
if 'logistic_results'    not in st.session_state: st.session_state.logistic_results = []
if 'recodings'           not in st.session_state: st.session_state.recodings = []
if 'cleaning_ops'        not in st.session_state: st.session_state.cleaning_ops = []  # [{cols, ops}]
if 'results'             not in st.session_state: st.session_state.results = {'czestosci': {}, 'krzyzowe': {}, 'srednie': {}, 'opisowe': {}, 'korelacje': {}}
if 'chi_results'         not in st.session_state: st.session_state.chi_results = {}
if 'custom_missing'      not in st.session_state: st.session_state.custom_missing = {}
if 'weights'             not in st.session_state: st.session_state.weights = None
if 'weight_targets'      not in st.session_state: st.session_state.weight_targets = {}
if 'treat_empty_as_miss' not in st.session_state: st.session_state.treat_empty_as_miss = False
if 'regression_results'  not in st.session_state: st.session_state.regression_results = []
if 'anova_results'       not in st.session_state: st.session_state.anova_results = []
if 'factor_results'      not in st.session_state: st.session_state.factor_results = []
if 'reg_blocks'          not in st.session_state: st.session_state.reg_blocks = [[]]
if 'conjoint_results'    not in st.session_state: st.session_state.conjoint_results = []
if 'maxdiff_results'     not in st.session_state: st.session_state.maxdiff_results = []
if 'normality_results'   not in st.session_state: st.session_state.normality_results = {}
if 'wordcloud_results'   not in st.session_state: st.session_state.wordcloud_results = []
if 'split_var'           not in st.session_state: st.session_state.split_var = None
if 'maxdiff_pairs'       not in st.session_state: st.session_state.maxdiff_pairs = [('', '')]
if 'data_source'         not in st.session_state: st.session_state.data_source = 'spss'
if 'excel_col_types'     not in st.session_state: st.session_state.excel_col_types = {}
if 'excel_sheet'         not in st.session_state: st.session_state.excel_sheet = None

# \u2500\u2500 Helper: cumulative add/replace for list-based results \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
def _merge_result(results_list, new_entry, key_fn):
    """Append new_entry or replace existing with same key. Mutates list in place."""
    new_key = key_fn(new_entry)
    for i, existing in enumerate(results_list):
        if key_fn(existing) == new_key:
            results_list[i] = new_entry
            return
    results_list.append(new_entry)

# -- Helper: split file iteration (SPSS-style) ------------------------
def _iter_split_groups(df, df_raw, var_labels, split_var, weights=None):
    """Yield (group_label, df_slice, df_raw_slice, weights_slice) for each
    category of the split variable. If split_var is None/empty, yields
    a single item with the full dataset and group_label=''.

    group_label is used as a prefix/marker in result titles, e.g.:
      '' (no split) or 'plec=Kobieta' (with split).
    """
    if not split_var or split_var not in df_raw.columns:
        yield ('', df, df_raw, weights)
        return
    # Get unique values from df (which has labels applied)
    series_labeled = df[split_var] if split_var in df.columns else df_raw[split_var]
    # Drop NaN and get sorted unique categories
    try:
        unique_vals = sorted(series_labeled.dropna().unique(), key=lambda x: str(x))
    except Exception:
        unique_vals = list(series_labeled.dropna().unique())

    split_lbl = var_labels.get(split_var, split_var)
    for val in unique_vals:
        mask = (series_labeled == val)
        if mask.sum() == 0:
            continue
        df_slice     = df.loc[mask]
        df_raw_slice = df_raw.loc[mask]
        w_slice      = None
        if weights is not None:
            w_ser = pd.Series(weights, index=df_raw.index)
            w_slice = w_ser.loc[mask].values
        lbl = f"{split_lbl}={val}"
        yield (lbl, df_slice, df_raw_slice, w_slice)

def _split_badge(grp_label):
    """Render colored badge info that a result comes from a split group.
    Returns nothing if grp_label is empty."""
    if not grp_label:
        return
    st.markdown(
        f'<div style="display:inline-block;background:#FFF4CE;'
        f'border-left:3px solid #D97706;padding:4px 10px;margin:4px 0 8px 0;'
        f'border-radius:4px;font-size:0.88em;">'
        f'<strong>\U0001f500 Podzia\u0142:</strong> <code>{grp_label}</code>'
        f'</div>',
        unsafe_allow_html=True
    )

def _extract_split_from_title(title_key):
    """Extract (base_title, group_label) from a key like 'Q1 | plec=Kobieta'.
    Returns (title_key, '') if no split present."""
    if not isinstance(title_key, str):
        return title_key, ''
    if " | " in title_key:
        _base, _grp = title_key.rsplit(" | ", 1)
        if "=" in _grp:
            return _base, _grp
    return title_key, ''


def _is_count_col(col):
    """Czy kolumna to liczebnosc N (kolumna zaokraglana do liczb calkowitych)."""
    cs = str(col)
    cl = cs.lower()
    if "%" in cl or "procent" in cl:
        return False
    if cs in ("N", "Liczebnosc [N]", "Liczebno\u015b\u0107 [N]"):
        return True
    if "[n]" in cl:
        return True
    if cl == "n" or cl.startswith("liczebno") or cl.startswith("liczba"):
        return True
    return False


_EXPORT_SPECIAL_ROWS = {
    "Suma", "Baza (N) / Suma (%)",
    "Braki danych", "Braki danych (wykluczone z tabeli)",
    "Og\u00f3\u0142em (Wa\u017cne)",
}


def prepare_export_table(df, drop_empty_cats=False):
    """Przygotuj tabele wynikowa do eksportu do pliku.

    - zaokraglij kolumny liczebnosci [N] do liczb calkowitych (zawsze),
    - gdy drop_empty_cats=True: usun kategorie bez odpowiedzi (wiersze, a w
      tabelach krzyzowych takze kolumny, ktorych laczna baza N wynosi 0).
    Pracuje na kopii. Wierszy podsumowan (Suma, Braki, Baza) ani 'boxow' [..]
    nigdy nie usuwa. Bezpieczny: przy bledzie oddaje wersje tylko-zaokraglona.
    """
    if df is None or not hasattr(df, "columns"):
        return df
    d = df.copy()
    n_cols = [c for c in d.columns if _is_count_col(c)]

    def _round_int(x):
        if isinstance(x, bool):
            return x
        if isinstance(x, (int, np.integer)):
            return int(x)
        if isinstance(x, (float, np.floating)) and pd.notnull(x):
            return int(round(x))
        return x

    for c in n_cols:
        d[c] = d[c].map(_round_int)

    if not drop_empty_cats or not n_cols:
        return d

    def _is_special_row(idx):
        s = str(idx)
        return s in _EXPORT_SPECIAL_ROWS or s.startswith("[")

    def _num_or_none(v):
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float, np.integer, np.floating)) and pd.notnull(v):
            return float(v)
        return None

    try:
        # --- usun puste wiersze (kategorie bez odpowiedzi) ---
        keep_mask = []
        for idx, row in d.iterrows():
            if _is_special_row(idx):
                keep_mask.append(True)
                continue
            nums = [x for x in (_num_or_none(row[c]) for c in n_cols) if x is not None]
            keep_mask.append((not nums) or any(abs(x) > 0 for x in nums))
        if not all(keep_mask):
            d = d.loc[keep_mask]

        # --- usun puste kolumny (tabele krzyzowe: grupy N/% z baza 0) ---
        def _col_base(c):
            s = str(c)
            return s.rsplit(" [", 1)[0] if " [" in s else s

        non_special_pos = [p for p, idx in enumerate(d.index)
                           if not _is_special_row(idx)]
        groups = {}
        for j, c in enumerate(d.columns):
            groups.setdefault(_col_base(c), []).append((j, c))
        drop_cols = []
        for base, cols in groups.items():
            if base == "Suma":
                continue
            gn = [(j, c) for (j, c) in cols if _is_count_col(c)]
            if not gn:
                continue
            total = 0.0
            seen = False
            for (j, c) in gn:
                colvals = d.iloc[:, j]
                for p in non_special_pos:
                    x = _num_or_none(colvals.iloc[p])
                    if x is not None:
                        seen = True
                        total += abs(x)
            if seen and total == 0:
                drop_cols.extend([c for (j, c) in cols])
        if drop_cols:
            d = d.drop(columns=drop_cols)
    except Exception:
        d2 = df.copy()
        for c in n_cols:
            d2[c] = d2[c].map(_round_int)
        return d2
    return d


# =====================================================================
# POROWNANIE FAL BADANIA (wave-over-wave comparison)
# =====================================================================
def ztest_two_props(p1, n1, p2, n2):
    """Z-test dla roznicy dwoch proporcji z niezaleznych prob (np. dwie fale).

    p1, p2 -- proporcje w PROCENTACH (0-100); n1, n2 -- wielkosci prob.
    Zwraca z-score (float) albo np.nan, gdy test jest niemozliwy.
    Test: z = (x1-x2)/sqrt(p_pool*(1-p_pool)*(1/n1+1/n2)), gdzie x = p/100.
    """
    try:
        n1 = float(n1); n2 = float(n2)
        if n1 <= 0 or n2 <= 0:
            return float('nan')
        x1 = float(p1) / 100.0
        x2 = float(p2) / 100.0
        if not (0.0 <= x1 <= 1.0) or not (0.0 <= x2 <= 1.0):
            return float('nan')
        p_pool = (x1 * n1 + x2 * n2) / (n1 + n2)
        denom = p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2)
        if denom <= 0:
            return float('nan')
        return (x1 - x2) / (denom ** 0.5)
    except (TypeError, ValueError, ZeroDivisionError):
        return float('nan')


_WAVE_SPECIAL_ROWS = ('Suma', 'Og\u00f3\u0142em (Wa\u017cne)', 'Braki danych')


def _wave_extract_pct_n(freq_df):
    """Z tabeli czestosci (kolumny '... [N]' i '... [%]') wyciaga
    (pct_series, n_series, base_n). Pomija wiersze podsumowania/brakow.
    base_n = liczebnosc bazy ('Suma' lub 'Og\u00f3\u0142em (Wa\u017cne)'),
    a w razie ich braku -- suma N kategorii (bez box-setow '[...]')."""
    _empty = (pd.Series(dtype=float), pd.Series(dtype=float), float('nan'))
    if freq_df is None or not isinstance(freq_df, pd.DataFrame) or freq_df.empty:
        return _empty
    n_col = pct_col = None
    for c in freq_df.columns:
        cs = str(c)
        if '[N]' in cs and n_col is None:
            n_col = c
        if '[%]' in cs and pct_col is None:
            pct_col = c
    if n_col is None or pct_col is None:
        return _empty
    df2 = freq_df.copy()
    df2.index = [str(i) for i in df2.index]
    base_n = float('nan')
    for _br in ('Suma', 'Og\u00f3\u0142em (Wa\u017cne)'):
        if _br in df2.index:
            try:
                base_n = float(df2.loc[_br, n_col])
            except (TypeError, ValueError):
                base_n = float('nan')
            break
    cats = [i for i in df2.index if i not in _WAVE_SPECIAL_ROWS]
    pct_s = pd.to_numeric(df2.loc[cats, pct_col], errors='coerce')
    n_s = pd.to_numeric(df2.loc[cats, n_col], errors='coerce')
    pct_s.index = cats
    n_s.index = cats
    if base_n != base_n:  # NaN -> fallback: suma N kategorii bez box-setow
        _nonbox = [c for c in cats if not str(c).startswith('[')]
        try:
            base_n = float(n_s.reindex(_nonbox).sum())
        except Exception:
            base_n = float('nan')
    return pct_s, n_s, base_n


def build_wave_comparison(wave_freqs, wave_labels, do_sig=True):
    """Buduje strukture porownania JEDNEJ zmiennej miedzy falami.

    wave_freqs  -- lista DataFrame'ow czestosci (po jednym na fale; None=brak),
    wave_labels -- lista etykiet fal (ta sama dlugosc; etykiety MUSZA byc unikalne).

    Zwraca dict:
      'pct'   -- DataFrame (kategorie x fale) z procentami,
      'n'     -- DataFrame (kategorie x fale) z liczebnosciami,
      'base'  -- Series (fala -> baza N),
      'pairs' -- lista (falaA, falaB) kolejnych par,
      'pair_labels' -- ['A \u2192 B', ...],
      'delta' -- DataFrame (kategorie x para) roznica pp (B-A),
      'sig'   -- DataFrame (kategorie x para) w {-1,0,1}: istotny spadek/brak/wzrost.
    """
    labels = list(wave_labels)
    cats_order, seen, extracted = [], set(), []
    for fd in wave_freqs:
        ps, ns, bn = _wave_extract_pct_n(fd)
        extracted.append((ps, ns, bn))
        for c in ps.index:
            if c not in seen:
                seen.add(c)
                cats_order.append(c)
    pct_df = pd.DataFrame(index=cats_order, columns=labels, dtype=float)
    n_df = pd.DataFrame(index=cats_order, columns=labels, dtype=float)
    base = pd.Series(index=labels, dtype=float)
    for li, lab in enumerate(labels):
        ps, ns, bn = extracted[li]
        base[lab] = bn
        for c in ps.index:
            pct_df.loc[c, lab] = ps[c]
            n_df.loc[c, lab] = ns[c]
    pairs = [(labels[i], labels[i + 1]) for i in range(len(labels) - 1)]
    delta_df = pd.DataFrame(index=cats_order, dtype=float)
    sig_df = pd.DataFrame(index=cats_order, dtype=float)
    pair_labels = []
    for (a, b) in pairs:
        pair_lbl = a + ' \u2192 ' + b
        pair_labels.append(pair_lbl)
        d = pct_df[b] - pct_df[a]
        delta_df[pair_lbl] = d
        sig_col = []
        for c in cats_order:
            val = d.get(c)
            if val is None or val != val:
                sig_col.append(0)
                continue
            if do_sig:
                z = ztest_two_props(pct_df.loc[c, a], base[a],
                                    pct_df.loc[c, b], base[b])
                if z == z and abs(z) > 1.959963985:
                    sig_col.append(1 if val > 0 else -1)
                else:
                    sig_col.append(0)
            else:
                sig_col.append(0)
        sig_df[pair_lbl] = sig_col
    return {
        'pct': pct_df, 'n': n_df, 'base': base,
        'pairs': pairs, 'pair_labels': pair_labels,
        'delta': delta_df, 'sig': sig_df,
    }


__all__ = [
    'load_spss_data',
    'ExcelMeta',
    'load_excel_data',
    'load_csv_data',
    'sniff_csv_dialect',
    'auto_detect_mrs',
    'auto_detect_matrix',
    '_apply_value_order',
    'build_matrix_table',
    'apply_segmentations',
    'apply_recodings',
    'apply_hclust_columns',
    'apply_cleaning_ops',
    'get_var_display_name',
    'get_weighted_stats',
    'apply_means_sig_testing',
    'apply_sig_testing',
    'build_banner_table',
    'build_banner_table_multi',
    'parse_banner_blocks',
    'ztest_two_props',
    'build_wave_comparison',
    'module_header',
    '_format_means_table',
    'get_streamlit_format',
    'safe_style',
    '_to_float_pct',
    '_fmt_cell',
    '_style_p',
    '_style_vif',
    '_style_loading',
    '_color_pair_row',
    '_color_sig',
    '_make_style_matrix_row',
    '_make_color_corr_cell',
    '_make_style_md',
    'calculate_rim_weights',
    'calculate_correlations',
    'run_regression_block',
    'run_anova',
    'run_factor_analysis',
    'safe_excel_val',
    'export_toc_sheet',
    'export_tables_to_sheet',
    'export_regression_to_excel',
    'export_anova_to_excel',
    'export_factor_to_excel',
    'export_matrix_to_excel',
    'write_db_sheet',
    'export_db_to_excel',
    'run_conjoint_rating',
    'run_conjoint_cbc',
    'export_conjoint_to_excel',
    'run_maxdiff',
    'export_maxdiff_to_excel',
    '_merge_result',
    '_iter_split_groups',
    '_split_badge',
    '_extract_split_from_title',
    'prepare_export_table',
]
