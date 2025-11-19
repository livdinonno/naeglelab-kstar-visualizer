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
    except Exception:
        # Fallback: always let them grab an HTML version if image export fails
        html = pio.to_html(fig, include_plotlyjs="cdn").encode("utf-8")
        st.download_button(
            "Download HTML (fallback)",
            data=html,
            file_name=f"{base_filename}.html",
            mime="text/html",
        )
        st.info(
            "Image export for this format is not available in the current environment. "
            "An HTML version of the figure was provided instead."
        )

# Sidebar
with st.sidebar:
    st.subheader("Background Context")
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
    st.caption("Run KSTAR to generate kinase activity output.")
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
        <div style='background-color:#f5f5f5; padding: 1rem 1.25rem; border-radius: 8px; font-size:0.95rem;'>
          This page helps you explore kinase activity results produced by KSTAR.
          Upload the activity file and, optionally, the FPR file from a KSTAR run,
          and the app will summarize how kinase activity patterns change across your samples.
          Below, you can view an activity–FPR dot plot, a heatmap across samples,
          the cleaned data table, single-kinase views, and a simple two-group comparison.
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
    "Each dot is a kinase–sample pair. The activity **Score** summarizes how strongly KSTAR infers that a kinase is active in that sample (values closer to 1 are stronger signals). "
    "The **FPR** (false positive rate) is the probability that a score of that size could appear by chance. "
    "High score with low FPR means a strong and reliable signal for that kinase in that sample."
)

if "FPR" not in merged.columns or merged["FPR"].isna().all():
    st.info("Upload an FPR file to enable this section.")
else:
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        fpr_thr = st.number_input(
            "FPR threshold", min_value=0.0, max_value=1.0, value=0.05, step=0.01
        )
        st.caption("Dots with FPR ≤ threshold are red (significant); others are grey.")
    with c2:
        topn_conf = st.slider("Top N by confidence (0 = all)", 0, 5000, 0)
        st.caption(
            "Top N is applied to kinase–sample pairs by lowest FPR. "
            "If N is small, only the most confident kinase–sample pairs will be shown."
        )
    with c3:
        order_mode = st.radio(
            "Kinase ordering",
            [
                "No Sorting",
                "By Activity (Descending)",
                "By Activity (Ascending)",
                "Manual Reordering",
                "By Hierarchical Clustering",
            ],
            horizontal=True,
        )

    # how many kinases total
    total_kinases = merged["Kinase"].nunique()
    st.caption(f"Total kinases detected in activity file: {total_kinases}")

    custom_k_order, custom_s_order = None, None
    if order_mode == "Manual Reordering":
        t1, t2 = st.columns(2)
        with t1:
            custom_k_order = st.text_area(
                "Kinase order (comma separated)",
                placeholder="EGFR, ERBB2, SRC, ABL1",
            )
        with t2:
            custom_s_order = st.text_area(
                "Sample order (optional, comma separated)",
                placeholder="Sample1, Sample2, Sample3",
            )

    dot = merged.copy()
    dot["FPR"] = pd.to_numeric(dot["FPR"], errors="coerce")
    dot["FPR"] = dot["FPR"].clip(lower=1e-300)
    dot["Significant"] = np.where(dot["FPR"] <= fpr_thr, "Sig", "Not Sig")

    working = dot.copy()
    conf_score = -np.log10(working["FPR"])
    if topn_conf > 0 and len(working) > topn_conf:
        # keep top N lowest-FPR pairs (highest confidence)
        working = working.iloc[np.argsort(-conf_score)[:topn_conf]]

    file_k_order = list(pd.Index(merged["Kinase"]).drop_duplicates())
    file_s_order = list(pd.Index(merged["Sample"]).drop_duplicates())

    # default orders
    y_order = file_k_order
    x_order = file_s_order

    # mean activity for ordering
    mean_activity = (
        merged.groupby("Kinase")["Score"]
        .mean()
        .reindex(file_k_order)
    )

    if order_mode == "By Activity (Descending)":
        y_order = list(mean_activity.sort_values(ascending=False).index)
    elif order_mode == "By Activity (Ascending)":
        y_order = list(mean_activity.sort_values(ascending=True).index)
    elif order_mode == "Manual Reordering":
        def _parse(txt):
            return [t.strip() for t in txt.split(",") if t.strip()]

        if custom_k_order:
            ordering = _parse(custom_k_order)
            ordering = [k for k in ordering if k in file_k_order]
            missing = [k for k in file_k_order if k not in ordering]
            y_order = ordering + missing
        if custom_s_order:
            ordering = _parse(custom_s_order)
            ordering = [s for s in ordering if s in file_s_order]
            missing = [s for s in file_s_order if s not in ordering]
            x_order = ordering + missing
    elif order_mode == "By Hierarchical Clustering":
        pivot = working.pivot_table(
            index="Kinase", columns="Sample", values="Score", aggfunc="mean"
        ).fillna(0)
        if pivot.shape[0] >= 3 and pivot.shape[1] >= 2:
            r_order, c_order = _cluster_order(pivot)
            y_order, x_order = r_order, c_order

    # sizes + colors (smaller than before to reduce overlap)
    nl = (-np.log10(working["FPR"])).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    bins = [0, 1, 2, 3, np.inf]
    labels = [6, 10, 14, 18]  # smaller dots → less visual pile-up
    idx = np.digitize(nl, bins) - 1
    size_vals = np.array(labels)[np.clip(idx, 0, len(labels) - 1)]
    colors = np.where(working["Significant"] == "Sig", "#d62728", "#d3d3d3")

    fig_dot = go.Figure()

    # invisible dummy trace to force all kinases onto the axis
    if len(x_order) > 0:
        fig_dot.add_trace(
            go.Scatter(
                x=[x_order[0]] * len(y_order),
                y=y_order,
                mode="markers",
                marker=dict(size=0, color="rgba(0,0,0,0)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # real data points
    fig_dot.add_trace(
        go.Scatter(
            x=working["Sample"],
            y=working["Kinase"],
            mode="markers",
            marker=dict(size=size_vals, color=colors, line=dict(width=0), opacity=0.9),
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

    # legend entries for significance
    fig_dot.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=10, color="#d62728"),
            showlegend=True,
            name=f"Significant (FPR ≤ {fpr_thr:g})",
        )
    )
    fig_dot.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=10, color="#d3d3d3"),
            showlegend=True,
            name=f"Not significant (FPR > {fpr_thr:g})",
        )
    )

    # dynamic height to space kinases more clearly
    plot_height = min(1600, max(800, 22 * len(y_order) + 220))

    fig_dot.update_layout(
        margin=dict(l=160, r=40, t=30, b=40),
        height=plot_height,
        legend_title_text="",
    )
    fig_dot.update_yaxes(
        categoryorder="array",
        categoryarray=y_order,
        tickfont=dict(size=10),
    )
    fig_dot.update_xaxes(
        categoryorder="array",
        categoryarray=x_order,
        tickangle=-45 if len(x_order) > 6 else 0,
    )

    plot_col, key_col = st.columns([5, 1])
    with plot_col:
        st.plotly_chart(fig_dot, use_container_width=True)
        st.caption(
            "A tall stack of red points for one kinase–sample combination means that kinase is consistently inferred active there at low FPR. "
            "Set **Top N by confidence** to 0 to show all kinase–sample pairs, including grey (higher FPR) points."
        )
    with key_col:
        st.markdown(
            """
            <div style='font-size:0.85rem; border:1px solid #dddddd; padding:0.5rem 0.75rem; border-radius:6px; background-color:#fafafa;'>
              <b>Dot size key</b><br>
              <span style='font-size:0.8rem;'>Relative confidence from FPR</span>
              <div style='margin-top:0.4rem; line-height:1.4;'>
                6&nbsp;px &nbsp;•&nbsp; highest FPR<br>
                10&nbsp;px •&nbsp; moderate FPR<br>
                14&nbsp;px •&nbsp; low FPR<br>
                18&nbsp;px •&nbsp; lowest FPR<br>
              </div>
              <div style='margin-top:0.4rem; font-size:0.8rem;'>
                Larger dots = lower FPR (more confident kinase–sample pairs).
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    fig_download_controls(fig_dot, "kstar_activity_fpr_dotplot", "dotplot_dl")

# 2) Activity Heatmap
st.divider()
st.subheader("2) Activity Heatmap")
st.markdown(
    "This heatmap shows how each kinase's activity score changes across samples. "
    "For each kinase, we first compute the mean and standard deviation of its activity across samples, "
    "then convert each value to a **z-score**: the number of standard deviations above or below that kinase's mean. "
    "Red cells indicate samples where that kinase is more active than its own average, and blue cells indicate lower-than-average activity."
)

mat = merged.pivot_table(
    index="Kinase", columns="Sample", values="Score", aggfunc="mean"
)

cluster_heatmap = st.checkbox("Cluster kinases and samples", value=False)
if cluster_heatmap and mat.shape[0] >= 3 and mat.shape[1] >= 2:
    try:
        r_order, c_order = _cluster_order(mat.fillna(mat.mean(axis=1)))
        mat = mat.loc[r_order, c_order]
    except Exception as e:
        st.warning(
            f"Clustering failed for this dataset, showing unclustered heatmap instead. Details: {e}"
        )

# z-score per kinase
means = mat.mean(axis=1, skipna=True).values.reshape(-1, 1)
stds = mat.std(axis=1, ddof=1, skipna=True).replace(0, np.nan).values.reshape(-1, 1)
zmat = (mat - means) / stds

sort_by = st.selectbox("Row ordering", ["None", "Row variance", "Row mean"])
if sort_by == "Row variance":
    zmat = zmat.loc[zmat.var(axis=1, skipna=True).sort_values(ascending=False).index]
elif sort_by == "Row mean":
    zmat = zmat.loc[
        zmat.mean(axis=1, skipna=True).sort_values(ascending=False).index
    ]

topn = st.slider("Top N most variable kinases", 0, max(10, zmat.shape[0]), 0)
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
fig_hm.update_layout(margin=dict(l=140, r=20, t=30, b=40), height=900)
st.plotly_chart(fig_hm, use_container_width=True)
fig_download_controls(fig_hm, "kstar_activity_heatmap", "heatmap_dl")

# 3) Data preview
st.divider()
st.subheader("3) Data Preview")
st.markdown(
    "This table shows the merged long-format KSTAR output used for all plots above. "
    "Each row is a kinase–sample pair with its activity score and, if provided, FPR. "
    "You can download this table and use it for your own downstream analysis."
)
left, right = st.columns([2, 1])
with left:
    st.dataframe(merged.head(50), use_container_width=True)
with right:
    st.metric("Kinases", merged["Kinase"].nunique())
    st.metric("Samples", merged["Sample"].nunique())
    st.metric("Total rows", len(merged))
st.download_button(
    "Download cleaned long-format table (CSV)",
    data=merged.to_csv(index=False).encode("utf-8"),
    file_name="kstar_long_table.csv",
    mime="text/csv",
)

# 4) Kinase detail
st.divider()
st.subheader("4) Kinase Detail")
st.markdown(
    "Use this section to focus on a single kinase. The plot shows that kinase's activity scores across samples. "
    "Look for samples where the score is noticeably higher or lower than the rest; those are samples where this kinase is unusually active or inactive."
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
fig_box.update_layout(margin=dict(l=40, r=20, t=40, b=40), height=450)
st.plotly_chart(fig_box, use_container_width=True)
fig_download_controls(fig_box, f"{sel_kinase}_detail", "kinase_detail_dl")
st.download_button(
    f"Download {sel_kinase} rows (CSV)",
    data=subk.to_csv(index=False).encode("utf-8"),
    file_name=f"{sel_kinase}_rows.csv",
    mime="text/csv",
)




