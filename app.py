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

# Sidebar
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

# Main page title
st.markdown("<h1 style='text-align:center; margin-top:0;'>KSTAR Results Plotter</h1>", unsafe_allow_html=True)

# Main intro (keeps your wording, removes italics)
with st.container():
    st.markdown(
        """
        <div style='background-color:#f5f5f5; padding: 1rem 1.25rem; border-radius: 8px;'>
          KSTAR predicts which kinases are most likely active in your samples by testing
          phosphosite evidence against kinase–substrate relationships. The output includes an activity score and a
          false positive rate (FPR) for each kinase sample.<br><br>
          How to read it: Higher activity score = stronger evidence a kinase is active. Lower FPR = higher confidence
          (values near 0 mean unlikely to be a false hit). Scores are visualized as colors and confidence by dot size (−log10(FPR)).
        </div>
        """,
        unsafe_allow_html=True
    )

# Uploads
colA, colB = st.columns(2)
with colA:
    file_activity = st.file_uploader("KSTAR ACTIVITIES FILE (.tsv)", type=["tsv"], key="file1_main")
with colB:
    file_fpr = st.file_uploader("KSTAR FPR (FALSE POSITIVE RATE) FILE (.tsv)", type=["tsv"], key="file2_main")

if not file_activity:
    st.info("Upload the KSTAR Activities file to begin. Add the FPR file to enable confidence sizing and the dot plot.")
    st.stop()

# Read and merge
act_raw = pd.read_csv(file_activity, sep="\t")
activity = coerce_activity(act_raw)
merged = activity.copy()
if file_fpr is not None:
    fpr_raw = pd.read_csv(file_fpr, sep="\t")
    fpr = coerce_fpr(fpr_raw)
    merged = pd.merge(activity, fpr, on=["Kinase", "Sample"], how="left")

# 1) Data preview
st.divider()
st.subheader("1) Data Preview and What It Means")
st.caption(
    "This table is what the app uses everywhere. Columns: Kinase, Sample, Score (activity), FPR (confidence). "
    "If FPR is missing in your file, the plots that use confidence will prompt you to add it."
)
c1, c2 = st.columns([2, 1])
with c1:
    st.dataframe(merged.head(50), use_container_width=True)
with c2:
    st.metric("Kinases", merged["Kinase"].nunique())
    st.metric("Samples", merged["Sample"].nunique())
    st.metric("Total rows", len(merged))
download_csv_button(merged, "Download cleaned long-format table", "kstar_long_table.csv")

# 2A) Heatmap
st.divider()
st.subheader("2A) Activity Heatmap")
st.markdown(
    "- What this shows: kinases on rows, samples on columns. Color is a z-score of the activity within each kinase so you can compare patterns across samples.\n"
    "- Why it matters: it quickly highlights kinases that shift up or down in certain conditions.\n"
    "- How to use it: reorder rows by variance or mean, and optionally limit to top-N most variable kinases to focus on strong patterns."
)
mat = merged.pivot_table(index="Kinase", columns="Sample", values="Score", aggfunc="mean")
zmat = (mat - mat.mean(axis=1, skipna=True).values.reshape(-1,1)) / mat.std(axis=1, ddof=1, skipna=True).values.reshape(-1,1)
sort_by = st.selectbox("Reorder rows by:", ["None", "Row variance", "Row mean"])
if sort_by == "Row variance":
    zmat = zmat.loc[zmat.var(axis=1, skipna=True).sort_values(ascending=False).index]
elif sort_by == "Row mean":
    zmat = zmat.loc[zmat.mean(axis=1, skipna=True).sort_values(ascending=False).index]
topn = st.slider("Show top N most variable kinases (0 = all)", 0, max(10, zmat.shape[0]), 0)
zplot = zmat.loc[zmat.var(axis=1, skipna=True).sort_values(ascending=False).index[:topn]] if (topn > 0 and zmat.shape[0] > topn) else zmat
fig_hm = px.imshow(zplot, labels=dict(color="Z-score"), aspect="auto", color_continuous_scale="RdBu", origin="lower")
fig_hm.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=600)
st.plotly_chart(fig_hm, use_container_width=True)

# 2B) Dot plot: Activity vs FPR
st.divider()
st.subheader("2B) Activity vs FPR Dot Plot")
st.markdown(
    "- What this shows: each dot is one kinase in one sample. Color is the activity score. Dot size reflects confidence as −log10(FPR); bigger = more confident.\n"
    "- Why it matters: lets you see, in one view, which kinases look active and how confident that call is.\n"
    "- How to use it: set a maximum FPR to hide low-confidence hits and optionally limit to the top results by confidence."
)
if "FPR" not in merged.columns or merged["FPR"].isna().all():
    st.info("Add an FPR file to enable confidence sizing and this dot plot.")
else:
    dot = merged.copy()
    dot["FPR"] = pd.to_numeric(dot["FPR"], errors="coerce")
    dot["neglog10FPR"] = -np.log10(dot["FPR"].clip(lower=1e-300))
    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        fpr_thr = st.number_input("Max FPR", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
    with c2:
        topn_conf = st.slider("Top-N by confidence (0 = all)", 0, 2000, 0)
    filt = dot[dot["FPR"] <= fpr_thr].copy()
    if topn_conf > 0 and len(filt) > topn_conf:
        filt = filt.sort_values("neglog10FPR", ascending=False).head(topn_conf)
    fig_dot = px.scatter(
        filt, x="Sample", y="Kinase", color="Score", size="neglog10FPR",
        size_max=22, color_continuous_scale="Viridis",
        hover_data={"FPR":":.3g","neglog10FPR":":.2f","Score":":.3g","Sample":True,"Kinase":True},
        title="Activity (color) vs Confidence (dot size = −log10(FPR))"
    )
    fig_dot.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=600)
    st.plotly_chart(fig_dot, use_container_width=True)

# 3) Kinase detail
st.divider()
st.subheader("3) Kinase Detail (per-sample view)")
st.markdown(
    "- What this shows: activity values for a single kinase across all samples.\n"
    "- Why it matters: confirms whether a kinase is consistently higher or lower in certain conditions.\n"
    "- How to use it: select a kinase, scan the spread of points and the box summary."
)
sel_kinase = st.selectbox("Choose a kinase", sorted(merged["Kinase"].unique()))
subk = merged[merged["Kinase"] == sel_kinase].copy()
fig_box = px.box(subk, x="Sample", y="Score", points="all", title=f"{sel_kinase} activity per sample")
fig_box.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=450)
st.plotly_chart(fig_box, use_container_width=True)
download_csv_button(subk, f"Download {sel_kinase} rows", f"{sel_kinase}_rows.csv")

# 4) Differential analysis
st.divider()
st.subheader("4) Differential Analysis (two groups)")
st.markdown(
    "- What this shows: for each kinase, the difference in mean activity between two groups of samples and how statistically convincing that difference is.\n"
    "- Why it matters: prioritizes kinases that change the most between conditions.\n"
    "- How to use it: label samples into two groups below. The volcano plot shows effect size (x) versus significance (y = −log10 FDR)."
)
samples = sorted(merged["Sample"].unique())
default_groups = ensure_groups_from_metadata(samples)
editable = pd.DataFrame({"Sample": samples, "Group": [default_groups[s] for s in samples]})
st.markdown("Edit group labels below. There must be exactly two unique group names.")
group_df = st.data_editor(editable, use_container_width=True, hide_index=True)
group_map = dict(zip(group_df["Sample"], group_df["Group"]))
try:
    diff = compute_group_stats(merged[["Kinase","Sample","Score"]], group_map)
    plot_df = diff.copy()
    plot_df["neglog10FDR"] = -np.log10(plot_df["FDR"].astype(float))
    fig_volc = px.scatter(
        plot_df, x="Diff_B_minus_A", y="neglog10FDR",
        hover_data=["Kinase","MeanA","MeanB","PValue","FDR","N_A","N_B"],
        title="Volcano: difference (B − A) vs −log10(FDR)"
    )
    fig_volc.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=500)
    st.plotly_chart(fig_volc, use_container_width=True)
    st.markdown("Filter significant kinases")
    fdr_thr = st.slider("FDR threshold", 0.0, 0.25, 0.05, 0.01)
    absdiff_thr_default = float(np.nanmax(np.abs(plot_df["Diff_B_minus_A"])) if len(plot_df) else 1.0)
    absdiff_thr = st.slider("Absolute difference threshold", 0.0, absdiff_thr_default, 0.0, 0.1)
    filt = diff[(diff["FDR"] <= fdr_thr) & (diff["Diff_B_minus_A"].abs() >= absdiff_thr)].sort_values("FDR")
    st.dataframe(filt, use_container_width=True, height=350)
    download_csv_button(diff, "Download all differential stats (CSV)", "kstar_diff_stats.csv")
    if len(filt):
        download_csv_button(filt, "Download filtered significant kinases (CSV)", "kstar_diff_significant.csv")
except Exception as e:
    st.info(f"Set exactly two group labels to run the analysis. Details: {e}")
