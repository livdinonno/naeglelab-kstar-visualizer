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

def _std_colname(name):
    return str(name).strip().lower()

def coerce_kstar_schema(df: pd.DataFrame):
    """
    Standardize to long format with columns:
    Kinase, Sample, Score, (optional) PValue, FDR
    Accepts long or wide inputs.
    """
    lower_cols = {_std_colname(c): c for c in df.columns}

    # Long-format candidates
    for k_col, s_col, v_col in [
        ("kinase", "sample", "score"),
        ("kinase", "sample", "activity"),
        ("kinase", "sample", "value"),
    ]:
        if k_col in lower_cols and s_col in lower_cols and v_col in lower_cols:
            out = df.rename(
                columns={
                    lower_cols[k_col]: "Kinase",
                    lower_cols[s_col]: "Sample",
                    lower_cols[v_col]: "Score",
                }
            ).copy()
            if "pvalue" in lower_cols:
                out.rename(columns={lower_cols["pvalue"]: "PValue"}, inplace=True)
            if "fdr" in lower_cols:
                out.rename(columns={lower_cols["fdr"]: "FDR"}, inplace=True)
            out["Score"] = pd.to_numeric(out["Score"], errors="coerce")
            return out

    # Otherwise treat as wide: first column is Kinase, rest are samples
    kinase_like = None
    for c in df.columns:
        if _std_colname(c) in ("kinase", "gene", "protein", "name", "id"):
            kinase_like = c
            break
    if kinase_like is None:
        kinase_like = df.columns[0]

    wide = df.rename(columns={kinase_like: "Kinase"}).copy()
    long_df = wide.melt(id_vars="Kinase", var_name="Sample", value_name="Score")
    long_df["Score"] = pd.to_numeric(long_df["Score"], errors="coerce")
    return long_df

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
    # External Links with better captions
    st.subheader("External Links")
    st.caption("Another way to run KSTAR to generate kinase activity output (External Runner).")
    st.markdown(f"- [{KSTAR_URL}]({KSTAR_URL})")
    st.caption("Step-by-step tutorial on how to use GitHub.")
    st.markdown(f"- [{TUTORIAL_URL}]({TUTORIAL_URL})")

    st.markdown("---")
    
    st.subheader("Further Context")
    st.caption("See key publications related to KSTAR methodology and applications.")
    with st.expander("Related Publications", expanded=False):
        publications = [
            {
                "title": "Phosphotyrosine Profiling Reveals New Signaling Networks (PMC Article)",
                "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4974343/"
            },
        ]
        for pub in publications:
            st.markdown(f"- [{pub['title']}]({pub['url']})")

    

    


# MAIN: Results Plotter 
st.markdown(
    "<h1 style='text-align:center; margin-top:0;'>KSTAR Results Plotter</h1>",
    unsafe_allow_html=True
)

with st.container():
    st.markdown(
        """
        <div style='background-color:#f5f5f5; padding: 1rem 1.25rem; border-radius: 8px;'>
          Upload your KSTAR output files below (TSV). The app will standardize columns and enable heatmaps,
          per-kinase drilldowns, and a simple two-group differential analysis.
        </div>
        """,
        unsafe_allow_html=True
    )

# Two TSV uploads on the MAIN page only
colA, colB = st.columns(2)
with colA:
    file1 = st.file_uploader(" KSTAR ACTIVITIES FILE (.tsv)", type=["tsv"], key="file1_main")
with colB:
    file2 = st.file_uploader("KSTAR FPR (FALSE POSITIVE RATE) FILE (.tsv) — optional", type=["tsv"], key="file2_main")

if not file1 and not file2:
    st.info("Upload at least one TSV file to begin.")
else:
    # Read/standardize each file, then concatenate if both are provided
    dfs = []
    for f in [file1, file2]:
        if f is None:
            continue
        raw = pd.read_csv(f, sep="\t")
        dfs.append(coerce_kstar_schema(raw))
    data = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]

    data["Kinase"] = data["Kinase"].astype(str)
    data["Sample"] = data["Sample"].astype(str)

    st.divider()
    st.subheader("1) Data Preview & Summary")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.dataframe(data.head(50), use_container_width=True)
    with c2:
        st.metric("Kinases", data["Kinase"].nunique())
        st.metric("Samples", data["Sample"].nunique())
        st.metric("Total rows", len(data))
    download_csv_button(data, "Download cleaned long-format table", "kstar_long_table.csv")

    # Heatmap
    st.divider()
    st.subheader("2) Activity Heatmap")
    st.caption("Row-wise Z-score per kinase to highlight patterns across samples.")
    mat = data.pivot_table(index="Kinase", columns="Sample", values="Score", aggfunc="mean")
    zmat = (mat - mat.mean(axis=1, skipna=True).values.reshape(-1,1)) / mat.std(axis=1, ddof=1, skipna=True).values.reshape(-1,1)

    sort_by = st.selectbox("Reorder rows by:", ["None", "Row variance", "Row mean"])
    if sort_by == "Row variance":
        zmat = zmat.loc[zmat.var(axis=1, skipna=True).sort_values(ascending=False).index]
    elif sort_by == "Row mean":
        zmat = zmat.loc[zmat.mean(axis=1, skipna=True).sort_values(ascending=False).index]

    topn = st.slider("Show top N most variable kinases (0 = all)", 0, max(10, zmat.shape[0]), 0)
    zplot = zmat.loc[zmat.var(axis=1, skipna=True).sort_values(ascending=False).index[:topn]] if (topn > 0 and zmat.shape[0] > topn) else zmat

    fig_hm = px.imshow(
        zplot,
        labels=dict(color="Z-score"),
        aspect="auto",
        color_continuous_scale="RdBu",
        origin="lower",
    )
    fig_hm.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=600)
    st.plotly_chart(fig_hm, use_container_width=True)

    # Per-kinase drilldown
    st.divider()
    st.subheader("3) Kinase Detail")
    sel_kinase = st.selectbox("Choose a kinase", sorted(data["Kinase"].unique()))
    subk = data[data["Kinase"] == sel_kinase].copy()
    fig_box = px.box(subk, x="Sample", y="Score", points="all", title=f"{sel_kinase} activity per sample")
    fig_box.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=450)
    st.plotly_chart(fig_box, use_container_width=True)
    download_csv_button(subk, f"Download {sel_kinase} rows", f"{sel_kinase}_rows.csv")

    # Differential analysis
    st.divider()
    st.subheader("4) Differential Analysis (two groups)")
    samples = sorted(data["Sample"].unique())
    default_groups = ensure_groups_from_metadata(samples)
    st.caption("Assign each sample to one of two groups. A heuristic pre-fills by name prefix.")
    editable = pd.DataFrame({"Sample": samples, "Group": [default_groups[s] for s in samples]})
    st.markdown("Edit groups below if needed (exactly two unique group names required).")
    group_df = st.data_editor(editable, use_container_width=True, hide_index=True)
    group_map = dict(zip(group_df["Sample"], group_df["Group"]))

    try:
        diff = compute_group_stats(data[["Kinase", "Sample", "Score"]], group_map)
        plot_df = diff.copy()
        plot_df["neglog10FDR"] = -np.log10(plot_df["FDR"].astype(float))
        fig_volc = px.scatter(
            plot_df,
            x="Diff_B_minus_A",
            y="neglog10FDR",
            hover_data=["Kinase", "MeanA", "MeanB", "PValue", "FDR", "N_A", "N_B"],
            title="Volcano: difference (B − A) vs −log10(FDR)"
        )
        fig_volc.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=500)
        st.plotly_chart(fig_volc, use_container_width=True)

        st.markdown("**Filter significant kinases**")
        fdr_thr = st.slider("FDR threshold", 0.0, 0.25, 0.05, 0.01)
        absdiff_thr_default = float(np.nanmax(np.abs(plot_df["Diff_B_minus_A"])) if len(plot_df) else 1.0)
        absdiff_thr = st.slider("Absolute difference threshold", 0.0, absdiff_thr_default, 0.0, 0.1)
        filt = diff[(diff["FDR"] <= fdr_thr) & (diff["Diff_B_minus_A"].abs() >= absdiff_thr)].sort_values("FDR", ascending=True)
        st.dataframe(filt, use_container_width=True, height=350)
        download_csv_button(diff, "Download all differential stats (CSV)", "kstar_diff_stats.csv")
        if len(filt):
            download_csv_button(filt, "Download filtered significant kinases (CSV)", "kstar_diff_significant.csv")
    except Exception as e:
        st.info(f"Set exactly two group labels to run the analysis. Details: {e}")


    
