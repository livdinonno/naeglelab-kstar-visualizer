import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from scipy import stats
from scipy.cluster.hierarchy import linkage, leaves_list
from statsmodels.stats.multitest import multipletests
from io import BytesIO
from PIL import Image

TUTORIAL_URL = "https://docs.github.com/en/get-started/start-your-journey/hello-world"
KSTAR_URL = "https://naeglelab-test-proteome-scout-3.pods.uvarc.io/kstar/"

st.set_page_config(page_title="KSTAR Results Plotter", layout="wide")

# Helpers
def _std(s):
    return str(s).strip().lower()

def _melt_wide(df, value_name):
    first = df.columns[0]
    df = df.rename(columns={first: "Kinase"}).copy()
    return df.melt(id_vars="Kinase", var_name="Sample", value_name=value_name)

def coerce_activity(df):
    cols = {_std(c): c for c in df.columns}
    for k_col, s_col, v_col in [
        ("kinase", "sample", "score"),
        ("kinase", "sample", "activity"),
        ("kinase", "sample", "value"),
    ]:
        if k_col in cols and s_col in cols and v_col in cols:
            out = df.rename(
                columns={
                    cols[k_col]: "Kinase",
                    cols[s_col]: "Sample",
                    cols[v_col]: "Score",
                }
            ).copy()
            if "pvalue" in cols:
                out.rename(columns={cols["pvalue"]: "PValue"}, inplace=True)
            if "fdr" in cols:
                out.rename(columns={cols["fdr"]: "FDR"}, inplace=True)
            out["Score"] = pd.to_numeric(out["Score"], errors="coerce")
            out["Kinase"] = out["Kinase"].astype(str)
            out["Sample"] = out["Sample"].astype(str)
            return out
    out = _melt_wide(df, "Score")
    out["Score"] = pd.to_numeric(out["Score"], errors="coerce")
    out["Kinase"] = out["Kinase"].astype(str)
    out["Sample"] = out["Sample"].astype(str)
    return out

def coerce_fpr(df):
    cols = {_std(c): c for c in df.columns}
    if "kinase" in cols and "sample" in cols and ("fpr" in cols or "false positive rate" in cols):
        vcol = cols["fpr"] if "fpr" in cols else cols["false positive rate"]
        out = df.rename(
            columns={
                cols["kinase"]: "Kinase",
                cols["sample"]: "Sample",
                vcol: "FPR",
            }
        ).copy()
    else:
        out = _melt_wide(df, "FPR")
    out["FPR"] = pd.to_numeric(out["FPR"], errors="coerce")
    out["Kinase"] = out["Kinase"].astype(str)
    out["Sample"] = out["Sample"].astype(str)
    return out

def ensure_groups_from_metadata(samples):
    groups = {}
    for s in samples:
        token = None
        for sep in ["_", "-"]:
            if sep in s:
                token = s.split(sep)[0]
                break
        groups[s] = token if token else "Group1"
    return groups

def compute_group_stats(df_long, group_map):
    work = df_long.copy()
    work["Group"] = work["Sample"].map(group_map)
    groups = sorted([g for g in pd.unique(work["Group"]) if pd.notna(g)])
    if len(groups) != 2:
        raise ValueError("Please define exactly two groups for differential analysis.")
    g1, g2 = groups
    rows = []
    for kinase, sub in work.groupby("Kinase", dropna=False):
        s1 = sub.loc[sub["Group"] == g1, "Score"].dropna().values
        s2 = sub.loc[sub["Group"] == g2, "Score"].dropna().values
        if len(s1) >= 2 and len(s2) >= 2:
            t_stat, p = stats.ttest_ind(s1, s2, equal_var=False, nan_policy="omit")
        else:
            t_stat, p = np.nan, np.nan
        rows.append(
            dict(
                Kinase=kinase,
                GroupA=g1,
                MeanA=(np.nanmean(s1) if len(s1) else np.nan),
                N_A=len(s1),
                GroupB=g2,
                MeanB=(np.nanmean(s2) if len(s2) else np.nan),
                N_B=len(s2),
                Diff_B_minus_A=(

                    (np.nanmean(s2) if len(s2) else np.nan)
                    - (np.nanmean(s1) if len(s1) else np.nan)
                ),
                T=t_stat,
                PValue=p,
            )
        )
    res = pd.DataFrame(rows)
    if "PValue" in res.columns:
        pvals = res["PValue"].values
        mask = np.isfinite(pvals)
        fdr = np.full_like(pvals, np.nan, dtype=float)
        if mask.sum() > 0:
            fdr[mask] = multipletests(pvals[mask], method="fdr_bh")[1]
        res["FDR"] = fdr
    return res

def _cluster_order(matrix_df):
    # cluster rows
    row_valid = matrix_df.fillna(matrix_df.mean(axis=1))
    try:
        row_link = linkage(row_valid.values, method="average", metric="euclidean")
        row_order = row_valid.index[leaves_list(row_link)]
    except Exception:
        row_order = matrix_df.index
    # cluster columns
    col_valid = matrix_df.T.fillna(matrix_df.T.mean(axis=1))
    try:
        col_link = linkage(col_valid.values, method="average", metric="euclidean")
        col_order = col_valid.columns[leaves_list(col_link)]
    except Exception:
        col_order = matrix_df.columns
    return list(row_order), list(col_order)

def fig_download_controls(fig, base_filename, key_prefix):
    fmt = st.selectbox(
        "Type of file to save as:",
        [".png", ".jpg", ".pdf", ".svg", ".eps", ".tif"],
        key=f"{key_prefix}_fmt",
    )

    data = None
    file_name = None
    mime = None
    label = "Download"

    try:
        if fmt in [".png", ".jpg", ".svg", ".pdf"]:
            data = pio.to_image(fig, format=fmt.replace(".", ""), scale=2)
            file_name = f"{base_filename}{fmt}"
            mime = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".svg": "image/svg+xml",
                ".pdf": "application/pdf",
            }[fmt]
        else:
            png = pio.to_image(fig, format="png", scale=2)
            im = Image.open(BytesIO(png)).convert("RGB")
            bio = BytesIO()
            if fmt == ".tif":
                im.save(bio, format="TIFF", compression="tiff_lzw")
                data = bio.getvalue()
                file_name = f"{base_filename}.tif"
                mime = "image/tiff"
            else:
                im.save(bio, format="EPS")
                data = bio.getvalue()
                file_name = f"{base_filename}.eps"
                mime = "application/postscript"
    except Exception:
        # Fallback: always let them grab an HTML version if image export fails
        data = pio.to_html(fig, include_plotlyjs="cdn").encode("utf-8")
        file_name = f"{base_filename}.html"
        mime = "text/html"
        label = "Download HTML (fallback)"
        st.info(
            "Image export for this format is not available in the current environment. "
            "An HTML version of the figure was provided instead."
        )

    st.download_button(
        label,
        data=data,
        file_name=file_name,
        mime=mime,
        key=f"{key_prefix}_download_button",
    )
