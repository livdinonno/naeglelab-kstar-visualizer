import streamlit as st
import pandas as pd
import numpy as np
import io
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
    Accepts either long or wide tables and returns a standardized long table:
    columns: Kinase, Sample, Score, (optional) PValue, FDR
    """
    # Try to detect long vs wide
    lower_cols = {_std_colname(c): c for c in df.columns}

    # Long format detection: must have at least Kinase/Sample/Score-ish columns
    long_candidates = [
        ("kinase", "sample", "score"),
        ("kinase", "sample", "activity"),
        ("kinase", "sample", "value"),
    ]
    for k_col, s_col, v_col in long_candidates:
        if k_col in lower_cols and s_col in lower_cols and v_col in lower_cols:
            out = df.rename(
                columns={
                    lower_cols[k_col]: "Kinase",
                    lower_cols[s_col]: "Sample",
                    lower_cols[v_col]: "Score",
                }
            ).copy()
            # Optional stats
            if "pvalue" in lower_cols:
                out.rename(columns={lower_cols["pvalue"]: "PValue"}, inplace=True)
            if "fdr" in lower_cols:
                out.rename(columns={lower_cols["fdr"]: "FDR"}, inplace=True)
            return out

    # Otherwise assume wide: first column = Kinase, rest = samples
    # Look for a column that looks like kinase
    kinase_like = None
    for c in df.columns:
        if _std_colname(c) in ("kinase", "gene", "protein", "name", "id"):
            kinase_like = c
            break
    if kinase_like is None:
        # fallback to first column
        kinase_like = df.columns[0]

    wide = df.rename(columns={kinase_like: "Kinase"}).copy()
    value_cols = [c for c in wide.columns if c != "Kinase"]
    long_df = wide.melt(id_vars="Kinase", var_name="Sample", value_name="Score")

    # Clean types
    long_df["Score"] = pd.to_numeric(long_df["Score"], errors="coerce")
    return long_df

def ensure_groups_from_metadata(samples):
    """
    Provide a default grouping from sample names if none supplied.
    Heuristic: split on first '_' or '-' token; fall back to 'Group1'.
    """
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
    """
    Returns a table with per-kinase mean by group, diff, t-test, and FDR.
    df_long: Kinase, Sample, Score
    group_map: dict Sample->Group (exactly two groups for this simple analysis)
    """
    work = df_long.copy()
    work["Group"] = work["Sample"].map(group_map)
    groups = sorted([g for g in pd.unique(work["Group"]) if pd.notna(g)])

    if len(groups) != 2:
        raise ValueError("Please define exactly two groups for differential analysis.")

    g1, g2 = groups
    # Pivot to have replicates per group
    # We'll compute t-test across sample replicates for each kinase
    results = []
    for kinase, sub in work.groupby("Kinase", dropna=False):
        s1 = sub.loc[sub["Group"] == g1, "Score"].dropna().values
        s2 = sub.loc[sub["Group"] == g2, "Score"].dropna().values
        if len(s1) >= 2 and len(s2) >= 2:
            t_stat, p = stats.ttest_ind(s1, s2, equal_var=False, nan_policy="omit")
        else:
            t_stat, p = np.nan, np.nan
        mean1 = np.nanmean(s1) if len(s1) else np.nan
        mean2 = np.nanmean(s2) if len(s2) else np.nan
        diff = mean2 - mean1
        results.append(
            dict(
                Kinase=kinase,
                GroupA=g1, MeanA=mean1, N_A=len(s1),
                GroupB=g2, MeanB=mean2, N_B=len(s2),
                Diff_B_minus_A=diff,
                T=t_stat, PValue=p
            )
        )
    res = pd.DataFrame(results)
    if "PValue" in res.columns:
        # FDR
        pvals = res["PValue"].values
        mask = np.isfinite(pvals)
        fdr = np.full_like(pvals, fill_value=np.nan, dtype=float)
        if mask.sum() > 0:
            fdr[mask] = multipletests(pvals[mask], method="fdr_bh")[1]
        res["FDR"] = fdr
    return res

def download_csv_button(df, label, filename):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, data=csv, file_name=filename, mime="text/csv")


# Sidebar
with st.sidebar:
    st.header("KSTAR Results Plotter")
    st.markdown("Upload KSTAR outputs (CSV/TSV) to explore kinase activity.")
    uploaded = st.file_uploader("Upload CSV or TSV", type=["csv", "tsv"])
    st.caption("Expected columns (long format): Kinase, Sample, Score. Optional: PValue, FDR.")
    st.caption("Wide format also accepted (Kinase as first column, samples as other columns).")

    st.markdown("---")
    st.markdown("**External Links**")
    st.markdown(f"- KSTAR Runner: [{KSTAR_URL}]({KSTAR_URL})")
    st.markdown(f"- Tutorial: [{TUTORIAL_URL}]({TUTORIAL_URL})")

# -----------------------------
# Landing section (gray card)
# -----------------------------
with st.container():
    st.markdown(
        """
        <div style='background-color:#f5f5f5; padding: 2rem; border-radius: 8px; text-align: center;'>
            <h1 style="margin-top:0;">Welcome to the KSTAR Results Plotter</h1>
            <p>This app helps visualize kinase activity outputs from KSTAR. Upload your results to view heatmaps,
            drill into single kinases, and compare groups with a simple differential analysis.</p>
            <a href="{kurl}" target="_blank"
               style="display:inline-block; background-color:#d8f3dc; color:#000; font-weight:600;
                      padding:0.75rem 1.5rem; border-radius:20px; text-decoration:none; border:1px solid #b7e4c7;">
               Run KSTAR
            </a>
        </div>
        """.format(kurl=KSTAR_URL),
        unsafe_allow_html=True
    )

st.markdown("### Ready to explore your data? Upload a file in the left sidebar.")

# Data loading
if uploaded is not None:
    # Detect sep
    sep = "\t" if Path(uploaded.name).suffix.lower() == ".tsv" else ","
    raw = pd.read_csv(uploaded, sep=sep)
    data = coerce_kstar_schema(raw)

    # Clean up obvious issues
    data["Kinase"] = data["Kinase"].astype(str)
    data["Sample"] = data["Sample"].astype(str)
    data["Score"] = pd.to_numeric(data["Score"], errors="coerce")

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
    st.caption("Z-score per kinase (row-wise) so patterns across samples are easier to compare.")

    # Pivot to matrix kinases x samples
    mat = data.pivot_table(index="Kinase", columns="Sample", values="Score", aggfunc="mean")
    # Z-score by row
    zmat = (mat - mat.mean(axis=1, skipna=True).values.reshape(-1,1)) / mat.std(axis=1, ddof=1, skipna=True).values.reshape(-1,1)

    # Ordering options
    sort_by_var = st.selectbox("Reorder rows by:", ["None", "Row variance", "Row mean"])
    if sort_by_var == "Row variance":
        zmat = zmat.loc[zmat.var(axis=1, skipna=True).sort_values(ascending=False).index]
    elif sort_by_var == "Row mean":
        zmat = zmat.loc[zmat.mean(axis=1, skipna=True).sort_values(ascending=False).index]

    # Optional top-N filter for readability
    topn = st.slider("Show top N most variable kinases (0 = show all)", 0, max(10, zmat.shape[0]), 0)
    if topn > 0 and zmat.shape[0] > topn:
        order = zmat.var(axis=1, skipna=True).sort_values(ascending=False).index[:topn]
        zplot = zmat.loc[order]
    else:
        zplot = zmat

    fig_hm = px.imshow(
        zplot,
        labels=dict(color="Z-score"),
        aspect="auto",
        color_continuous_scale="RdBu",
        origin="lower"
    )
    fig_hm.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=600)
    st.plotly_chart(fig_hm, use_container_width=True)


    # Per-kinase drilldown
    st.divider()
    st.subheader("3) Kinase Detail")
    sel_kinase = st.selectbox("Choose a kinase", sorted(data["Kinase"].unique()))
    subk = data[data["Kinase"] == sel_kinase].copy()

    # Box/strip plot per sample
    fig_box = px.box(subk, x="Sample", y="Score", points="all", title=f"{sel_kinase} activity per sample")
    fig_box.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=450)
    st.plotly_chart(fig_box, use_container_width=True)

    st.caption("Download rows for this kinase")
    download_csv_button(subk, f"Download {sel_kinase} rows", f"{sel_kinase}_rows.csv")


    # Differential analysis (simple)
    st.divider()
    st.subheader("4) Differential Analysis (two groups)")

    samples = sorted(data["Sample"].unique())
    default_groups = ensure_groups_from_metadata(samples)

    st.caption("Assign each sample to one of two groups. A quick heuristic pre-fills groups by sample name prefix.")
    editable = pd.DataFrame(
        {"Sample": samples, "Group": [default_groups[s] for s in samples]}
    )
    st.markdown("Edit groups below if needed (exactly two unique group names required).")
    group_df = st.data_editor(editable, use_container_width=True, hide_index=True)

    group_map = dict(zip(group_df["Sample"], group_df["Group"]))
    try:
        diff = compute_group_stats(data[["Kinase", "Sample", "Score"]], group_map)
        # Volcano plot
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

        # Filter table
        st.markdown("**Filter significant kinases**")
        fdr_thr = st.slider("FDR threshold", 0.0, 0.25, 0.05, 0.01)
        absdiff_thr = st.slider("Absolute difference threshold", 0.0, float(np.nanmax(np.abs(plot_df["Diff_B_minus_A"])) if len(plot_df)>0 else 1.0), 0.0, 0.1)
        filt = diff[
            (diff["FDR"] <= fdr_thr) &
            (diff["Diff_B_minus_A"].abs() >= absdiff_thr)
        ].sort_values("FDR", ascending=True)
        st.dataframe(filt, use_container_width=True, height=350)

        download_csv_button(diff, "Download all differential stats (CSV)", "kstar_diff_stats.csv")
        if len(filt):
            download_csv_button(filt, "Download filtered significant kinases (CSV)", "kstar_diff_significant.csv")

    except Exception as e:
        st.info(f"Set exactly two group labels to run the analysis. Details: {e}")

    # References / publications
    st.divider()
    st.subheader("5) Related Publications")
    st.caption("Key literature related to KSTAR methodology and applications:")
    pubs = [
        {
            "title": "Phosphotyrosine Profiling Reveals New Signaling Networks (PMC)",
            "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4974343/"
        },
    ]
    for p in pubs:
        st.markdown(f"- [{p['title']}]({p['url']})")

else:
    # No file yet: keep a helpful gray section extending past the caption
    st.divider()
    with st.container():
        st.caption("Click the button below to view the step-by-step tutorial on GitHub.")
        st.link_button("Open GitHub Tutorial", TUTORIAL_URL)
        st.caption("See key publications related to KSTAR methodology and applications.")
        with st.expander("Related Publications", expanded=False):
            st.markdown("- [Phosphotyrosine Profiling Reveals New Signaling Networks (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4974343/)")

