import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
from scipy import stats
from statsmodels.stats.multitest import multipletests


TUTORIAL_URL = "https://docs.github.com/en/get-started/start-your-journey/hello-world"
KSTAR_URL = "https://naeglelab-test-proteome-scout-3.pods.uvarc.io/kstar/"

st.set_page_config(page_title="KSTAR Results Plotter", layout="wide")


# Helpers
def _std(s): return str(s).strip().lower()

def _melt_wide(df, value_name):
    first = df.columns[0]
    df = df.rename(columns={first: "Kinase"}).copy()
    return df.melt(id_vars="Kinase", var_name="Sample", value_name=value_name)

def coerce_activity(df):
    cols = {_std(c): c for c in df.columns}
    for k_col, s_col, v_col in [("kinase","sample","score"), ("kinase","sample","activity"), ("kinase","sample","value")]:
        if k_col in cols and s_col in cols and v_col in cols:
            out = df.rename(columns={cols[k_col]:"Kinase", cols[s_col]:"Sample", cols[v_col]:"Score"}).copy()
            if "pvalue" in cols: out.rename(columns={cols["pvalue"]:"PValue"}, inplace=True)
            if "fdr" in cols: out.rename(columns={cols["fdr"]:"FDR"}, inplace=True)
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
        out = df.rename(columns={cols["kinase"]:"Kinase", cols["sample"]:"Sample", vcol:"FPR"}).copy()
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
        rows.append(dict(
            Kinase=kinase,
            GroupA=g1, MeanA=(np.nanmean(s1) if len(s1) else np.nan), N_A=len(s1),
            GroupB=g2, MeanB=(np.nanmean(s2) if len(s2) else np.nan), N_B=len(s2),
            Diff_B_minus_A=((np.nanmean(s2) if len(s2) else np.nan) - (np.nanmean(s1) if len(s1) else np.nan)),
            T=t_stat, PValue=p
        ))
    res = pd.DataFrame(rows)
    if "PValue" in res.columns:
        pvals = res["PValue"].values
        mask = np.isfinite(pvals)
        fdr = np.full_like(pvals, np.nan, dtype=float)
        if mask.sum() > 0:
            fdr[mask] = multipletests(pvals[mask], method="fdr_bh")[1]
        res["FDR"] = fdr
    return res

def download_csv_button(df, label, filename):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, data=csv, file_name=filename, mime="text/csv")


# SIDEBAR
with st.sidebar:
    st.subheader("Related Publications")
    with st.expander("Related Publications", expanded=False):
        publications = [
            {"title": "Phosphotyrosine Profiling Reveals New Signaling Networks (PMC Article)",
             "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4974343/"},
        ]
        for pub in publications:
            st.markdown(f"- [{pub['title']}]({pub['url']})")

    st.markdown("---")
    st.subheader("External Links")
    st.caption("Run KSTAR to generate kinase activity output (external runner).")
    st.markdown(f"- [{KSTAR_URL}]({KSTAR_URL})")
    st.caption("Step-by-step tutorial on GitHub.")
    st.markdown(f"- [{TUTORIAL_URL}]({TUTORIAL_URL})")


# MAIN PAGE
st.markdown("<h1 style='text-align:center; margin-top:0;'>KSTAR Results Plotter</h1>", unsafe_allow_html=True)

with st.container():
    st.markdown(
        """
        <div style='background-color:#f5f5f5; padding: 1rem 1.25rem; border-radius: 8px;'>
          <b>What am I looking at?</b> KSTAR predicts which kinases are most likely active in your samples by testing
          phosphosite evidence against kinase–substrate relationships. The output includes an <i>activity score</i> and a
          <i>false positive rate (FPR)</i> for each Kinase×Sample.<br><br>
          <b>How to read it:</b> Higher <i>activity score</i> ⇒ stronger evidence a kinase is active. Lower <i>FPR</i> ⇒ higher confidence
          (values near 0 mean unlikely to be a false hit). We visualize scores as colors and confidence by dot size (−log10(FPR)).
        </div>
        """,
        unsafe_allow_html=True
    )


# Upload section 
colA, colB = st.columns(2)
with colA:
    file_activity = st.file_uploader("KSTAR ACTIVITIES FILE (.tsv)", type=["tsv"], key="file1_main")
with colB:
    file_fpr = st.file_uploader("KSTAR FPR (FALSE POSITIVE RATE) FILE (.tsv)", type=["tsv"], key="file2_main")

if not file_activity:
    st.info("Upload the KSTAR Activities file (and optionally the FPR file) to begin.")
    st.stop()

# Read & merge
act_raw = pd.read_csv(file_activity, sep="\t")
activity = coerce_activity(act_raw)
merged = activity.copy()
if file_fpr is not None:
    fpr_raw = pd.read_csv(file_fpr, sep="\t")
    fpr = coerce_fpr(fpr_raw)
    merged = pd.merge(activity, fpr, on=["Kinase", "Sample"], how="left")

# Data Preview
st.divider()
st.subheader("1) Data Preview & Summary")
st.caption("Standardized long-format data used in all visualizations. Columns are `Kinase`



    
