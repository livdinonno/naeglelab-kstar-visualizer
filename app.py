import streamlit as st
import pandas as pd
import numpy as np
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
    row_valid = matrix_df.fillna(matrix_df.mean(axis=1))
    try:
        row_link = linkage(row_valid.values, method="average", metric="euclidean")
        row_order = row_valid.index[leaves_list(row_link)]
    except Exception:
        row_order = matrix_df.index
    col_valid = matrix_df.T.fillna(matrix_df.T.mean(axis=1))
    try:
        col_link = linkage(col_valid.values, method="average", metric="euclidean")
        col_order = matrix_df.columns[leaves_list(col_link)]
    except Exception:
        col_order = matrix_df.columns
    return list(row_order), list(col_order)

def fig_download_controls(fig, base_filename, key_prefix):
    fmt = st.selectbox(
        "Type of file to save as:",
        [".png", ".jpg", ".pdf", ".svg", ".eps", ".tif"],
        key=f"{key_prefix}_fmt",
    )
    btn = st.button("Download", key=f"{key_prefix}_dl")
    if not btn:
        return
    try:
        if fmt in [".png", ".jpg", ".svg", ".pdf"]:
            data = pio.to_image(fig, format=fmt.replace(".", ""), scale=2)
            st.download_button(
                "Save file",
                data=data,
                file_name=f"{base_filename}{fmt}",
                mime={
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".svg": "image/svg+xml",
                    ".pdf": "application/pdf",
                }[fmt],
            )
        else:
            from io import BytesIO
            from PIL import Image

            png = pio.to_image(fig, format="png", scale=2)
            im = Image.open(BytesIO(png)).convert("RGB")
            bio = BytesIO()
            if fmt == ".tif":
                im.save(bio, format="TIFF", compression="tiff_lzw")
                st.download_button(
                    "Save file",
                    data=bio.getvalue(),
                    file_name=f"{base_filename}.tif",
                    mime="image/tiff",
                )
            else:
                im.save(bio, format="EPS")
                st.download_button(
                    "Save file",
                    data=bio.getvalue(),
                    file_name=f"{base_filename}.eps",
                    mime="application/postscript",
                )
    except Exception as e:
        html = pio.to_html(fig, include_plotlyjs="cdn").encode("utf-8")
        st.download_button(
            "Download HTML (fallback)",
            data=html,
            file_name=f"{base_filename}.html",
            mime="text/html",
        )
        st.info(f"Could not generate {fmt}. Saved HTML instead. Details: {e}")

# Sidebar
with st.sidebar:
    st.subheader("Related Publications")
    with st.expander("Related Publications", expanded=False):
        publications = [
            {
                "title": "Phosphotyrosine Profiling Reveals New Signaling Networks (PMC Article)",
                "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4974343/",
            },
        ]
        for pub in publications:
            st.markdown(f"- [{pub['title']}]({pub['url']})")
    st.markdown("---")
    st.subheader("External Links")
    st.caption("Run KSTAR to generate kinase activity output (external runner).")
    st.markdown(f"- [{KSTAR_URL}]({KSTAR_URL})")
    st.caption("Step-by-step tutorial on GitHub.")
    st.markdown(f"- [{TUTORIAL_URL}]({TUTORIAL_URL})")

# Title and intro
st.markdown(
    "<h1 style='text-align:center; margin-top:0;'>KSTAR Results Plotter</h1>",
    unsafe_allow_html=True,
)
with st.container():
    st.markdown(
        """
        <div style='background-color:#f5f5f5; padding: 1rem 1.25rem; border-radius: 8px;'>
          KSTAR estimates which kinases are active in each sample using phosphosite evidence and known kinase–substrate relationships. 
          You will see two numbers: a score and a false positive rate (FPR). Both go from 0 to 1. A score near 1 suggests stronger evidence 
          the kinase is active in that sample, while a score near 0 suggests weaker evidence. A lower FPR means higher confidence in that call. 
          An FPR of 0.05 or lower is a common threshold for significance.
        </div>
        """,
        unsafe_allow_html=True,
    )

# Uploads
colA, colB = st.columns(2)
with colA:
    st.markdown(
        "<div style='font-size:1.05rem; font-weight:700; margin-bottom:0.25rem;'>KSTAR ACTIVITIES FILE (.tsv)</div>",
        unsafe_allow_html=True,
    )
    file_activity = st.file_uploader(
        "KSTAR ACTIVITIES FILE (.tsv)",
        type=["tsv"],
        key="file1_main",
        label_visibility="collapsed",
    )
with colB:
    st.markdown(
        "<div style='font-size:1.05rem; font-weight:700; margin-bottom:0.25rem;'>KSTAR FPR (FALSE POSITIVE RATE) FILE (.tsv)</div>",
        unsafe_allow_html=True,
    )
    file_fpr = st.file_uploader(
        "KSTAR FPR (FALSE POSITIVE RATE) FILE (.tsv)",
        type=["tsv"],
        key="file2_main",
        label_visibility="collapsed",
    )

if not file_activity:
    st.info(
        "Upload the KSTAR Activities file to begin. Add the FPR file to enable confidence coloring and the dot plot."
    )
    st.stop()

# Read and merge
act_raw = pd.read_csv(file_activity, sep="\t")
activity = coerce_activity(act_raw)
merged = activity.copy()
if file_fpr is not None:
    fpr_raw = pd.read_csv(file_fpr, sep="\t")
    fpr = coerce_fpr(fpr_raw)
    merged = pd.merge(activity, fpr, on=["Kinase", "Sample"], how="left")

# 1) Dot plot
st.divider()
st.subheader("1) Activity vs FPR Dot Plot")
st.markdown(
    "This plot shows one dot for each kinase in each sample. A red dot means it meets the FPR threshold (commonly 0.05); "
    "a light gray dot means it does not. Larger dots mean higher confidence (lower FPR). "
    "You can change the threshold, limit the view to the top results by confidence, and choose how to order the axes."
)

if "FPR" not in merged.columns or merged["FPR"].isna().all():
    st.info("Add an FPR file to enable this dot plot and the confidence sizing.")
else:
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        fpr_thr = st.number_input(
            "FPR threshold", min_value=0.0, max_value=1.0, value=0.05, step=0.01
        )
        st.caption("Points with FPR at or below this value are red (significant).")
    with c2:
        topn_conf = st.slider("Top N by confidence (0 = show all)", 0, 5000, 0)
        st.caption(
            "Shows only the N dots with the lowest FPR (highest confidence). Set 0 to show all dots."
        )
    with c3:
        order_mode = st.radio(
            "Axis order",
            ["Keep file order", "Alphabetical", "Cluster", "Custom"],
            horizontal=True,
        )
        st.caption(
            "Keep file order = original input order. Alphabetical = sort by name. "
            "Cluster = group by similarity. Custom = paste your own order."
        )

    custom_k_order, custom_s_order = None, None
    if order_mode == "Custom":
        t1, t2 = st.columns(2)
        with t1:
            custom_k_order = st.text_area(
                "Kinase order (comma separated)",
                placeholder="EGFR, ERBB2, ERBB3, EPHB2, SRC, ABL1",
            )
        with t2:
            custom_s_order = st.text_area(
                "Sample order (comma separated)",
                placeholder="Sample1, Sample2, Sample3, Sample4",
            )

    dot = merged.copy()
    dot["FPR"] = pd.to_numeric(dot["FPR"], errors="coerce")
    dot["FPR"] = dot["FPR"].clip(lower=1e-300)
    dot["Significant"] = np.where(dot["FPR"] <= fpr_thr, "Sig", "Not Sig")

    working = dot.copy()
    conf_score = -np.log10(working["FPR"])
    if topn_conf > 0 and len(working) > topn_conf:
        working = working.iloc[np.argsort(-conf_score)[:topn_conf]]

    file_k_order = list(pd.Index(merged["Kinase"]).drop_duplicates())
    file_s_order = list(pd.Index(merged["Sample"]).drop_duplicates())
    if order_mode == "Alphabetical":
        y_order = sorted(working["Kinase"].unique())
        x_order = sorted(working["Sample"].unique())
    elif order_mode == "Cluster":
        pivot = working.pivot_table(
            index="Kinase", columns="Sample", values="Score", aggfunc="mean"
        ).fillna(0)
        if pivot.shape[0] >= 3 and pivot.shape[1] >= 2:
            r_order, c_order = _cluster_order(pivot)
            y_order, x_order = r_order, c_order
        else:
            y_order, x_order = file_k_order, file_s_order
    elif order_mode == "Custom":
        def _parse_list(txt):
            return [t.strip() for t in txt.split(",") if t.strip()]
        y_order = _parse_list(custom_k_order) if custom_k_order else file_k_order
        x_order = _parse_list(custom_s_order) if custom_s_order else file_s_order
    else:
        y_order, x_order = file_k_order, file_s_order

    nl = (-np.log10(working["FPR"])).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    bins = [0, 1, 2, 3, np.inf]
    labels = [8, 16, 24, 32]
    size_idx = np.digitize(nl, bins) - 1
    size_vals = np.array(labels)[np.clip(size_idx, 0, len(labels) - 1)]
    colors = np.where(working["Significant"] == "Sig", "#d62728", "#d3d3d3")

    fig_dot = go.Figure()
    fig_dot.add_trace(
        go.Scatter(
            x=working["Sample"],
            y=working["Kinase"],
            mode="markers",
            marker=dict(size=size_vals, color=colors, line=dict(width=0)),
            text=[
                f"Kinase: {k}<br>Sample: {s}<br>Score: {sc:.3g}<br>FPR: {f:.3g}"
                for k, s, sc, f in zip(
                    working["Kinase"],
                    working["Sample"],
                    working["Score"],
                    working["FPR"],
                )
            ],
            hoverinfo="text",
            showlegend=False,
        )
    )
    fig_dot.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=16, color="#d62728"),
            showlegend=True,
            name=f"Significant (FPR ≤ {fpr_thr:g})",
        )
    )
    fig_dot.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=16, color="#d3d3d3"),
            showlegend=True,
            name=f"Not Significant (FPR > {fpr_thr:g})",
        )
    )
    fig_dot.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=1, color="rgba(0,0,0,0)"),
            showlegend=True,
            name="Higher confidence ↓",
        )
    )
    for s in [32, 24, 16, 8]:
        fig_dot.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=s, color="#d62728"),
                showlegend=True,
                name=f"size {s}",
            )
        )

    fig_dot.update_layout(
        title="Red = Significant; Gray = Not Significant. Larger dots = higher confidence (lower FPR).",
        margin=dict(l=0, r=0, t=40, b=0),
        height=620,
        legend_title_text="",
    )
    fig_dot.update_yaxes(categoryorder="array", categoryarray=y_order)
    fig_dot.update_xaxes(categoryorder="array", categoryarray=x_order)
    st.plotly_chart(fig_dot, use_container_width=True)
    fig_download_controls(fig_dot, "kstar_activity_fpr_dotplot", "dotplot_dl")

# 2) Heatmap
st.divider()
st.subheader("2) Activity Heatmap")
st.markdown(
    "Rows are kinases and columns are samples. The color shows how each sample compares to that kinase’s own average (a z-score). "
    "Values near 0 are close to average for that kinase. Positive values mean higher than average. Negative values mean lower than average. "
    "Reordering by row variance brings the most variable kinases to the top; reordering by row mean brings kinases with higher average activity to the top. "
    "Top N limits the view to only the N most variable kinases. You can enable clustering to group similar rows and columns."
)
mat = merged.pivot_table(
    index="Kinase", columns="Sample", values="Score", aggfunc="mean"
)
cluster_heatmap = st.checkbox("Cluster kinases and samples (heatmap)", value=False)
if cluster_heatmap and mat.shape[0] >= 3 and mat.shape[1] >= 2:
    r_order, c_order = _cluster_order(mat.fillna(mat.mean(axis=1)))
    mat = mat.loc[r_order, c_order]
zmat = (mat - mat.mean(axis=1, skipna=True).values.reshape(-1, 1)) / mat.std(
    axis=1, ddof=1, skipna=True
).values.reshape(-1, 1)
sort_by = st.selectbox("Reorder rows by:", ["None", "Row variance", "Row mean"])
if sort_by == "Row variance":
    zmat = zmat.loc[zmat.var(axis=1, skipna=True).sort_values(ascending=False).index]
elif sort_by == "Row mean":
    zmat = zmat.loc[
        zmat.mean(axis=1, skipna=True).sort_values(ascending=False).index
    ]
topn = st.slider("Top N most variable kinases (0 = show all)", 0, max(10, zmat.shape[0]), 0)
if topn == 0 or zmat.shape[0] <= topn:
    zplot = zmat
else:
    zplot = zmat.loc[
        zmat.var(axis=1, skipna=True).sort_values(ascending=False).index[:topn]
    ]
fig_hm = px.imshow(
    zplot,
    labels=dict(color="Z-score"),
    aspect="auto",
    color_continuous_scale="RdBu",
    origin="lower",
)
fig_hm.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=600)
st.plotly_chart(fig_hm, use_container_width=True)
fig_download_controls(fig_hm, "kstar_activity_heatmap", "heatmap_dl")

# 3) Data preview
st.divider()
st.subheader("3) Data Preview and What It Means")
left, right = st.columns([2, 1])
with left:
    st.markdown(
        "This is the standardized table used by all plots. Total rows is the number of kinase–sample pairs "
        "(for example, 850 means 850 specific kinase–sample combinations). Score goes from 0 to 1 and summarizes evidence "
        "that a kinase is active in a sample (closer to 1 is stronger). FPR goes from 0 to 1 and measures confidence "
        "(lower is better; values at or below 0.05 are commonly treated as significant)."
    )
    st.dataframe(merged.head(50), use_container_width=True)
with right:
    st.metric("Kinases", merged["Kinase"].nunique())
    st.metric("Samples", merged["Sample"].nunique())
    st.metric("Total rows", len(merged))
    st.markdown(
        """
        <div style="border:1px solid #ddd; border-radius:8px; padding:8px; margin-top:6px;">
          <div><b>Score</b>: 0 — weaker evidence &nbsp;&nbsp; | &nbsp;&nbsp; 1 — stronger evidence</div>
          <div><b>FPR</b>: 1 — lower confidence &nbsp;&nbsp; | &nbsp;&nbsp; 0 — higher confidence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
st.download_button(
    "Download cleaned long-format table (CSV)",
    data=merged.to_csv(index=False).encode("utf-8"),
    file_name="kstar_long_table.csv",
    mime="text/csv",
)

# 4) Kinase detail
st.divider()
st.subheader("4) Kinase Detail (per-sample view)")
st.markdown(
    "Pick a kinase to see its scores across samples. Dots show individual values and the box shows spread. "
    "Use this view to confirm whether the kinase stays high or low in the samples you care about."
)
sel_kinase = st.selectbox("Choose a kinase", sorted(merged["Kinase"].unique()))
subk = merged[merged["Kinase"] == sel_kinase].copy()
fig_box = px.box(
    subk,
    x="Sample",
    y="Score",
    points="all",
    title=f"{sel_kinase} activity per sample",
)
fig_box.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=450)
st.plotly_chart(fig_box, use_container_width=True)
fig_download_controls(fig_box, f"{sel_kinase}_detail", "kinase_detail_dl")
st.download_button(
    f"Download {sel_kinase} rows (CSV)",
    data=subk.to_csv(index=False).encode("utf-8"),
    file_name=f"{sel_kinase}_rows.csv",
    mime="text/csv",
)

# 5) Differential analysis
st.divider()
st.subheader("5) Differential Analysis (two groups)")
st.markdown(
    "This compares two groups of samples, for example Control versus Treated. For each kinase, the app computes the difference in average score "
    "between the two groups and runs a statistical test, then adjusts for multiple comparisons. "
    "The volcano plot shows effect size on the x-axis (Group B minus Group A) and statistical strength on the y-axis. "
    "Label samples into exactly two groups below. Each group should have at least two samples for stable results."
)
samples = sorted(merged["Sample"].unique())
default_groups = ensure_groups_from_metadata(samples)
editable = pd.DataFrame(
    {"Sample": samples, "Group": [default_groups[s] for s in samples]}
)
st.markdown(
    "Edit group labels below (two distinct group names, at least two samples in each)."
)
group_df = st.data_editor(editable, use_container_width=True, hide_index=True)
group_map = dict(zip(group_df["Sample"], group_df["Group"]))

try:
    diff = compute_group_stats(merged[["Kinase", "Sample", "Score"]], group_map)
    plot_df = diff.copy()
    plot_df["neglog10FDR"] = -np.log10(plot_df["FDR"].astype(float))
    fig_volc = px.scatter(
        plot_df,
        x="Diff_B_minus_A",
        y="neglog10FDR",
        hover_data=[
            "Kinase",
            "MeanA",
            "MeanB",
            "PValue",
            "FDR",
            "N_A",
            "N_B",
        ],
        title="Volcano: difference (B − A) vs statistical strength",
    )
    fig_volc.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=500)
    st.plotly_chart(fig_volc, use_container_width=True)
    fig_download_controls(fig_volc, "kstar_volcano_plot", "volcano_dl")

    st.markdown("You can filter the table to focus on important changes.")
    fdr_thr = st.slider("FDR threshold", 0.0, 0.25, 0.05, 0.01)
    absdiff_thr_default = float(
        np.nanmax(np.abs(plot_df["Diff_B_minus_A"])) if len(plot_df) else 1.0
    )
    absdiff_thr = st.slider(
        "Absolute difference threshold", 0.0, absdiff_thr_default, 0.0, 0.1
    )
    filt = diff[
        (diff["FDR"] <= fdr_thr) & (diff["Diff_B_minus_A"].abs() >= absdiff_thr)
    ].sort_values("FDR")
    st.dataframe(filt, use_container_width=True, height=350)
    st.download_button(
        "Download all differential stats (CSV)",
        data=diff.to_csv(index=False).encode("utf-8"),
        file_name="kstar_diff_stats.csv",
        mime="text/csv",
    )
    if len(filt):
        st.download_button(
            "Download filtered significant kinases (CSV)",
            data=filt.to_csv(index=False).encode("utf-8"),
            file_name="kstar_diff_significant.csv",
            mime="text/csv",
        )

except Exception as e:
    st.info(
        f"Set exactly two valid group labels with enough samples per group. Details: {e}"
    )



