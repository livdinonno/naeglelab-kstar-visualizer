import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from scipy import stats
from scipy.cluster.hierarchy import linkage, leaves_list
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

def _download_plot_buttons(fig, base_filename):
    try:
        png_bytes = pio.to_image(fig, format="png", scale=2)
        st.download_button("Download PNG", data=png_bytes, file_name=f"{base_filename}.png", mime="image/png")
    except Exception:
        html_bytes = pio.to_html(fig, include_plotlyjs="cdn").encode("utf-8")
        st.download_button("Download HTML", data=html_bytes, file_name=f"{base_filename}.html", mime="text/html")

def _cluster_order(matrix_df):
    # Rows (kinases)
    row_valid = matrix_df.fillna(matrix_df.mean(axis=1))
    try:
        row_link = linkage(row_valid.values, method="average", metric="euclidean")
        row_order = row_valid.index[leaves_list(row_link)]
    except Exception:
        row_order = matrix_df.index
    # Columns (samples)
    col_valid = matrix_df.T.fillna(matrix_df.T.mean(axis=1))
    try:
        col_link = linkage(col_valid.values, method="average", metric="euclidean")
        col_order = matrix_df.columns[leaves_list(col_link)]
    except Exception:
        col_order = matrix_df.columns
    return list(row_order), list(col_order)

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

# Title
st.markdown("<h1 style='text-align:center; margin-top:0;'>KSTAR Results Plotter</h1>", unsafe_allow_html=True)

# Intro 
with st.container():
    st.markdown(
        """
        <div style='background-color:#f5f5f5; padding: 1rem 1.25rem; border-radius: 8px;'>
          KSTAR estimates which kinases are active in each sample using phosphosite evidence and known kinase–substrate relationships. 
          The two numbers you will see are the activity score and the false positive rate (FPR). The activity score goes from 0 to 1. 
          A value near 1 suggests stronger evidence the kinase is active in that sample. A value near 0 suggests little or no evidence. 
          FPR also goes from 0 to 1. Lower FPR values mean higher confidence in the call. An FPR of 0.05 or lower is a common threshold for significance.
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

# 1) Data Preview and What It Means
st.divider()
st.subheader("1) Data Preview and What It Means")
st.markdown(
    "This table is the working dataset for all plots. **Total rows** is the number of kinase-sample pairs in your files "
    "(for example, 850 means there are 850 combinations of a specific kinase measured in a specific sample). "
    "**Score** goes from 0 to 1 and summarizes how strongly the data support that a kinase is active in a sample "
    "(closer to 1 means stronger evidence, closer to 0 means weaker evidence). "
    "**FPR** goes from 0 to 1 and measures confidence in that score "
    "(lower is better; values ≤ 0.05 are typically considered significant)."
)
c1, c2 = st.columns([2, 1])
with c1:
    st.dataframe(merged.head(50), use_container_width=True)
with c2:
    st.metric("Kinases", merged["Kinase"].nunique())
    st.metric("Samples", merged["Sample"].nunique())
    st.metric("Total rows", len(merged))
download_csv_button = lambda df, label, filename: st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=filename, mime="text/csv")
download_csv_button(merged, "Download cleaned long-format table (CSV)", "kstar_long_table.csv")

# 2A) Activity Heatmap
st.divider()
st.subheader("2A) Activity Heatmap")
st.markdown(
    "Rows are kinases and columns are samples. The color shows a z-score of the activity for each kinase across samples, "
    "so you can compare patterns sample-to-sample. A z-score near 0 means that sample is close to that kinase’s average. "
    "Positive values (often up to about +3) mean higher than that kinase’s average in that sample, and negative values "
    "(down to about −3) mean lower than average. Reordering rows by **row variance** brings the most variable kinases to the top; "
    "reordering by **row mean** brings kinases with higher average activity to the top. The **Top N most variable kinases** selector "
    "just limits the view to the N kinases with the largest variance, which helps you focus on the strongest patterns."
)
mat = merged.pivot_table(index="Kinase", columns="Sample", values="Score", aggfunc="mean")

# Optional clustering toggle for the heatmap
cluster_heatmap = st.checkbox("Cluster kinases and samples (heatmap)", value=False)
if cluster_heatmap and mat.shape[0] >= 3 and mat.shape[1] >= 2:
    r_order, c_order = _cluster_order(mat.fillna(mat.mean(axis=1)))
    mat = mat.loc[r_order, c_order]

# Z-score by row
zmat = (mat - mat.mean(axis=1, skipna=True).values.reshape(-1,1)) / mat.std(axis=1, ddof=1, skipna=True).values.reshape(-1,1)

sort_by = st.selectbox("Reorder rows by:", ["None", "Row variance", "Row mean"])
if sort_by == "Row variance":
    zmat = zmat.loc[zmat.var(axis=1, skipna=True).sort_values(ascending=False).index]
elif sort_by == "Row mean":
    zmat = zmat.loc[zmat.mean(axis=1, skipna=True).sort_values(ascending=False).index]

topn = st.slider("Top N most variable kinases (0 = show all)", 0, max(10, zmat.shape[0]), 0)
zplot = zmat
if topn > 0 and zmat.shape[0] > topn:
    order = zmat.var(axis=1, skipna=True).sort_values(ascending=False).index[:topn]
    zplot = zmat.loc[order]

fig_hm = px.imshow(
    zplot,
    labels=dict(color="Z-score"),
    aspect="auto",
    color_continuous_scale="RdBu",
    origin="lower",
)
fig_hm.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=600)
st.plotly_chart(fig_hm, use_container_width=True)
_download_plot_buttons(fig_hm, "kstar_activity_heatmap")

# 2B) Activity vs FPR Dot Plot
st.divider()
st.subheader("2B) Activity vs FPR Dot Plot")
st.markdown(
    "Each dot is one kinase in one sample. The dot is **red** if it meets the FPR threshold and **light gray** if it does not. "
    "Dot size shows confidence using −log10(FPR): bigger dots are more confident. By default the FPR threshold is 0.05, "
    "which is commonly used. You can change it if needed. The “Top N by confidence” control limits the view to the N highest-confidence dots."
)

if "FPR" not in merged.columns or merged["FPR"].isna().all():
    st.info("Add an FPR file to enable this dot plot and confidence sizing.")
else:
    dot = merged.copy()
    dot["FPR"] = pd.to_numeric(dot["FPR"], errors="coerce")
    dot["neglog10FPR"] = -np.log10(dot["FPR"].clip(lower=1e-300))

    left, mid, right = st.columns([1,1,2])
    with left:
        fpr_thr = st.number_input("FPR threshold", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
    with mid:
        topn_conf = st.slider("Top N by confidence (0 = all)", 0, 5000, 0)

    # Label significant vs not
    dot["Significant"] = np.where(dot["FPR"] <= fpr_thr, "Sig", "Not Sig")

    # Keep only top N by confidence if asked
    working = dot.copy()
    if topn_conf > 0 and len(working) > topn_conf:
        working = working.sort_values("neglog10FPR", ascending=False).head(topn_conf)

    # Map size (visual): keep within a readable range
    # size = 6 .. 32, proportional to neglog10FPR
    nl = working["neglog10FPR"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if nl.max() <= 0:
        size_vals = np.full_like(nl, 8, dtype=float)
    else:
        size_vals = 6 + (nl / nl.max()) * (32 - 6)

    colors = np.where(working["Significant"] == "Sig", "#d62728", "#d3d3d3")  # red vs light gray

    fig_dot = go.Figure()
    fig_dot.add_trace(go.Scatter(
        x=working["Sample"], y=working["Kinase"],
        mode="markers",
        marker=dict(size=size_vals, color=colors, line=dict(width=0)),
        text=[f"Kinase: {k}<br>Sample: {s}<br>Score: {sc:.3g}<br>FPR: {f:.3g}<br>-log10(FPR): {nlf:.2f}"
              for k,s,sc,f,nlf in zip(working["Kinase"], working["Sample"], working["Score"], working["FPR"], working["neglog10FPR"])],
        hoverinfo="text",
        name="",
        showlegend=False
    ))

    # Size legend (example sizes)
    for label, nl_target in [("−log10(FPR) ≈ 1", 1), ("≈ 2", 2), ("≈ 3", 3), ("≈ 4", 4)]:
        size_ref = 6 + (nl_target / max(nl.max(), 1e-6)) * (32 - 6)
        fig_dot.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=max(size_ref, 6), color="#d62728"),
            legendgroup="size", showlegend=True, name=label
        ))

    fig_dot.update_layout(
        title="Red = passes FPR threshold, Gray = above threshold; size = −log10(FPR)",
        margin=dict(l=0, r=0, t=40, b=0),
        height=600,
        legend_title_text="Size guide"
    )

    # Optional clustering on dot plot axes (order kinases and samples)
    cluster_dots = st.checkbox("Cluster kinases and samples (dot plot)", value=False)
    if cluster_dots:
        # Cluster by activity scores; use pivot
        pivot = working.pivot_table(index="Kinase", columns="Sample", values="Score", aggfunc="mean").fillna(0)
        if pivot.shape[0] >= 3 and pivot.shape[1] >= 2:
            r_order, c_order = _cluster_order(pivot)
            fig_dot.update_yaxes(categoryorder="array", categoryarray=r_order)
            fig_dot.update_xaxes(categoryorder="array", categoryarray=c_order)

    st.plotly_chart(fig_dot, use_container_width=True)
    _download_plot_buttons(fig_dot, "kstar_activity_fpr_dotplot")

# 3) Kinase Detail (per-sample view)
st.divider()
st.subheader("3) Kinase Detail (per-sample view)")
st.markdown(
    "Pick a kinase to see its activity values across all samples. Dots show the individual values and the box summarizes the spread. "
    "Use this view to confirm whether the kinase is consistently higher or lower in specific conditions."
)
sel_kinase = st.selectbox("Choose a kinase", sorted(merged["Kinase"].unique()))
subk = merged[merged["Kinase"] == sel_kinase].copy()
fig_box = px.box(subk, x="Sample", y="Score", points="all", title=f"{sel_kinase} activity per sample")
fig_box.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=450)
st.plotly_chart(fig_box, use_container_width=True)
try:
    png = pio.to_image(fig_box, format="png", scale=2)
    st.download_button("Download kinase detail (PNG)", data=png, file_name=f"{sel_kinase}_detail.png", mime="image/png")
except Exception:
    html = pio.to_html(fig_box, include_plotlyjs="cdn").encode("utf-8")
    st.download_button("Download kinase detail (HTML)", data=html, file_name=f"{sel_kinase}_detail.html", mime="text/html")
st.download_button(f"Download {sel_kinase} rows (CSV)", data=subk.to_csv(index=False).encode("utf-8"),
                   file_name=f"{sel_kinase}_rows.csv", mime="text/csv")

# 4) Differential Analysis (two groups)
st.divider()
st.subheader("4) Differential Analysis (two groups)")
st.markdown(
    "This compares two groups of samples (for example, Control versus Treated). For each kinase, we compute the difference in average activity "
    "between the two groups and run a Welch’s t-test. We then adjust p-values using the Benjamini–Hochberg method to control FDR. "
    "The volcano plot shows effect size on the x-axis (Group B minus Group A) and −log10(FDR) on the y-axis. "
    "To make this work, label samples into exactly two groups below. Each kinase needs enough replicates in both groups to run a test."
)
samples = sorted(merged["Sample"].unique())
default_groups = ensure_groups_from_metadata(samples)
editable = pd.DataFrame({"Sample": samples, "Group": [default_groups[s] for s in samples]})
st.markdown("Edit group labels below. There must be exactly two unique group names, with at least two samples per group for reliable testing.")
group_df = st.data_editor(editable, use_container_width=True, hide_index=True)
group_map = dict(zip(group_df["Sample"], group_df["Group"]))

try:
    diff = compute_group_stats(merged[["Kinase","Sample","Score"]], group_map)

    # Basic checks so interactions “do something”
    counts = pd.Series(group_map).value_counts()
    show_warn = False
    if len(counts.index) != 2:
        st.warning("Please ensure there are exactly two group names.")
        show_warn = True
    elif (counts < 2).any():
        st.warning("Each group should have at least two samples for stable statistics.")
        show_warn = True

    plot_df = diff.copy()
    plot_df["neglog10FDR"] = -np.log10(plot_df["FDR"].astype(float))
    fig_volc = px.scatter(
        plot_df,
        x="Diff_B_minus_A",
        y="neglog10FDR",
        hover_data=["Kinase","MeanA","MeanB","PValue","FDR","N_A","N_B"],
        title="Volcano: difference (B − A) vs −log10(FDR)"
    )
    fig_volc.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=500)
    st.plotly_chart(fig_volc, use_container_width=True)
    _download_plot_buttons(fig_volc, "kstar_volcano_plot")

    st.markdown("Filter the table to focus on important changes. Lower FDR values indicate stronger evidence for a difference.")
    fdr_thr = st.slider("FDR threshold", 0.0, 0.25, 0.05, 0.01)
    absdiff_thr_default = float(np.nanmax(np.abs(plot_df["Diff_B_minus_A"])) if len(plot_df) else 1.0)
    absdiff_thr = st.slider("Absolute difference threshold", 0.0, absdiff_thr_default, 0.0, 0.1)
    filt = diff[(diff["FDR"] <= fdr_thr) & (diff["Diff_B_minus_A"].abs() >= absdiff_thr)].sort_values("FDR")
    st.dataframe(filt, use_container_width=True, height=350)
    st.download_button("Download all differential stats (CSV)", data=diff.to_csv(index=False).encode("utf-8"),
                       file_name="kstar_diff_stats.csv", mime="text/csv")
    if len(filt):
        st.download_button("Download filtered significant kinases (CSV)", data=filt.to_csv(index=False).encode("utf-8"),
                           file_name="kstar_diff_significant.csv", mime="text/csv")

    if show_warn:
        st.info("Adjust group labels to meet the requirements and the statistics will update automatically.")

except Exception as e:
    st.info(f"Set exactly two valid group labels to run the analysis. Details: {e}")

