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
import os
import re
import hashlib

TUTORIAL_URL = "https://naeglelab.github.io/KSTAR/Tutorial/tutorial.html"
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
        col_order = col_valid.columns[leaves_list(col_link)]
    except Exception:
        col_order = matrix_df.columns
    return list(row_order), list(col_order)

def fig_download_controls(fig, base_filename, key_prefix):
    col1, col2 = st.columns(2)

    html_bytes = fig.to_html(include_plotlyjs="cdn").encode("utf-8")
    col1.download_button(
        "Download plot (HTML)",
        data=html_bytes,
        file_name=f"{base_filename}.html",
        mime="text/html",
        key=f"{key_prefix}_html",
    )

    try:
        png_bytes = fig.to_image(format="png", scale=2)
        col2.download_button(
            "Download plot (PNG)",
            data=png_bytes,
            file_name=f"{base_filename}.png",
            mime="image/png",
            key=f"{key_prefix}_png",
        )
    except Exception:
        col2.info("PNG export unavailable. Install kaleido to enable image downloads.")

# Multi-file helpers
def _detect_uploaded_kind(filename):
    name = os.path.basename(str(filename)).lower()
    if "activity" in name or "activities" in name:
        return "activity"
    if "fpr" in name:
        return "fpr"
    return None

def _extract_split_number(filename):
    name = os.path.basename(str(filename)).lower()
    patterns = [
        r"split[_\-]?(\d+)",
        r"[_\-](\d+)(?:\.[a-z0-9]+)?$",
    ]
    for pattern in patterns:
        m = re.search(pattern, name)
        if m:
            return int(m.group(1))
    return 10**9

def _read_uploaded_tsv(uploaded_file):
    uploaded_file.seek(0)
    return pd.read_csv(uploaded_file, sep="\t")

def _get_uploaded_file_bytes(uploaded_file):
    uploaded_file.seek(0)
    data = uploaded_file.read()
    uploaded_file.seek(0)
    return data

def _get_uploaded_file_hash(uploaded_file):
    data = _get_uploaded_file_bytes(uploaded_file)
    return hashlib.sha256(data).hexdigest()

def _normalize_match_key(filename):
    name = os.path.basename(str(filename)).lower()
    stem = os.path.splitext(name)[0]

    stem = re.sub(r"false[_\-\s]*positive[_\-\s]*rate", "", stem)
    stem = re.sub(r"[_\-\s]+", "_", stem).strip("_")

    stem = re.sub(r"_activities_(\d+)$", r"_\1", stem)
    stem = re.sub(r"_activity_(\d+)$", r"_\1", stem)
    stem = re.sub(r"_fpr_(\d+)$", r"_\1", stem)

    stem = re.sub(r"_activities$", "", stem)
    stem = re.sub(r"_activity$", "", stem)
    stem = re.sub(r"_fpr$", "", stem)

    stem = re.sub(r"_+", "_", stem).strip("_")

    return stem

def _find_duplicate_content(files):
    groups = {}
    for f in files:
        file_hash = _get_uploaded_file_hash(f)
        groups.setdefault(file_hash, []).append(f.name)
    return {k: v for k, v in groups.items() if len(v) > 1}

def _summarize_upload_alignment(activity_files, fpr_files):
    activity_files = activity_files if activity_files else []
    fpr_files = fpr_files if fpr_files else []

    activity_keys = {}
    for f in activity_files:
        activity_keys.setdefault(_normalize_match_key(f.name), []).append(f.name)

    fpr_keys = {}
    for f in fpr_files:
        fpr_keys.setdefault(_normalize_match_key(f.name), []).append(f.name)

    shared_keys = sorted(set(activity_keys.keys()) & set(fpr_keys.keys()))
    missing_fpr = {k: activity_keys[k] for k in activity_keys if k not in fpr_keys}
    missing_activity = {k: fpr_keys[k] for k in fpr_keys if k not in activity_keys}

    return {
        "shared_keys": shared_keys,
        "missing_fpr": missing_fpr,
        "missing_activity": missing_activity,
        "activity_duplicate_content": _find_duplicate_content(activity_files),
        "fpr_duplicate_content": _find_duplicate_content(fpr_files),
    }

def _deduplicate_uploaded_files(files, label):
    if not files:
        return []

    seen_hashes = set()
    deduped = []
    skipped = []

    for f in files:
        file_hash = _get_uploaded_file_hash(f)
        if file_hash in seen_hashes:
            skipped.append(f.name)
            continue
        seen_hashes.add(file_hash)
        deduped.append(f)

    if len(skipped) > 0:
        st.warning(
            f"{label}: true duplicate files with identical content were skipped: "
            + ", ".join(skipped[:10])
            + (" ..." if len(skipped) > 10 else "")
        )

    return deduped

def _merge_uploaded_wide_files(file_list, label):
    if file_list is None or len(file_list) == 0:
        return None

    sorted_files = sorted(file_list, key=lambda f: _extract_split_number(f.name))
    dfs = []
    seen_cols = set()

    for f in sorted_files:
        df = _read_uploaded_tsv(f)

        if df.shape[1] == 0:
            st.warning(f"{label}: {f.name} appears empty and was skipped.")
            continue

        first = df.columns[0]
        df = df.rename(columns={first: "Kinase"}).copy()
        df["Kinase"] = df["Kinase"].astype(str).str.strip()

        if df["Kinase"].duplicated().any():
            df = df.groupby("Kinase", as_index=False).first()

        keep_cols = ["Kinase"]
        dropped_cols = []

        for c in df.columns[1:]:
            if c not in seen_cols:
                keep_cols.append(c)
                seen_cols.add(c)
            else:
                dropped_cols.append(c)

        if len(dropped_cols) > 0:
            st.warning(
                f"{label}: duplicate sample columns were skipped from {f.name}: "
                + ", ".join(dropped_cols[:10])
                + (" ..." if len(dropped_cols) > 10 else "")
            )

        df = df[keep_cols].copy()
        dfs.append(df)

    if len(dfs) == 0:
        return None

    if len(dfs) == 1:
        return dfs[0]

    merged = dfs[0]
    for d in dfs[1:]:
        merged = pd.merge(merged, d, on="Kinase", how="outer")

    return merged

def _prepare_uploaded_input(file_input, label, expected_kind=None):
    if not file_input:
        return None

    files = file_input if isinstance(file_input, list) else [file_input]

    if expected_kind is not None:
        detected = []
        for f in files:
            kind = _detect_uploaded_kind(f.name)
            if kind is None:
                detected.append(f)
            elif kind == expected_kind:
                detected.append(f)
            else:
                st.warning(
                    f"{label}: skipped {f.name} because it looks like a {kind} file, not {expected_kind}."
                )
        files = detected

    if len(files) == 0:
        return None

    files = _deduplicate_uploaded_files(files, label)

    return _merge_uploaded_wide_files(files, label)

# Sidebar
with st.sidebar:
    st.subheader("Background Context")
    with st.expander("Related Publications", expanded=False):
        publications = [
            {
                "title": "Phosphotyrosine Profiling Reveals New Signaling Networks (PMC Article)",
                "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4974343/",
            },
            {
                "title": "Global Phosphoproteomics Study (RSC Molecular Omics)",
                "url": "https://pubs.rsc.org/en/content/articlelanding/2023/mo/d3mo00042g",
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
            <p> KSTAR is a kinase activity prediction algorithm. This site allows you to visualize your data after processing it through the KSTAR algorithm. Below, you can view an activity: an FPR dot plot, a heatmap across samples,
          the cleaned data table, and single-kinase views. This site also provides quick access to a setup tutorial and relevant publications.</p>
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
        accept_multiple_files=True,
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
        accept_multiple_files=True,
    )

if file_activity or file_fpr:
    upload_summary = _summarize_upload_alignment(file_activity, file_fpr)

    if len(upload_summary["activity_duplicate_content"]) > 0:
        st.warning("True duplicate activity files detected based on identical file contents.")
        for _, names in upload_summary["activity_duplicate_content"].items():
            st.write(names)

    if len(upload_summary["fpr_duplicate_content"]) > 0:
        st.warning("True duplicate FPR files detected based on identical file contents.")
        for _, names in upload_summary["fpr_duplicate_content"].items():
            st.write(names)

    if len(upload_summary["missing_fpr"]) > 0:
        st.warning("Some activity files are missing a matching FPR file.")
        for key, names in upload_summary["missing_fpr"].items():
            st.write(f"- Missing FPR match for activity key `{key}`:")
            st.write(names)

    if len(upload_summary["missing_activity"]) > 0:
        st.warning("Some FPR files are missing a matching activity file.")
        for key, names in upload_summary["missing_activity"].items():
            st.write(f"- Missing activity match for FPR key `{key}`:")
            st.write(names)

if not file_activity:
    st.info(
        "Upload the KSTAR Activities file to begin. Add the FPR file to enable confidence coloring and the dot plot."
    )
    st.stop()

# Read and merge
act_raw = _prepare_uploaded_input(file_activity, "Activities", expected_kind="activity")
if act_raw is None:
    st.error("No valid activity files were uploaded.")
    st.stop()

activity = coerce_activity(act_raw)

activity["Sample"] = activity["Sample"].astype(str).str.strip()
activity["Kinase"] = activity["Kinase"].astype(str).str.strip()

merged = activity.copy()
if file_fpr is not None:
    fpr_raw = _prepare_uploaded_input(file_fpr, "FPR", expected_kind="fpr")

    if fpr_raw is None:
        st.warning("No valid FPR files were uploaded. Continuing with activities only.")
    else:
        fpr = coerce_fpr(fpr_raw)

        fpr["Sample"] = fpr["Sample"].astype(str).str.strip()
        fpr["Kinase"] = fpr["Kinase"].astype(str).str.strip()

        activity_samples = sorted(activity["Sample"].dropna().unique())
        fpr_samples = sorted(fpr["Sample"].dropna().unique())

        missing_in_fpr = [s for s in activity_samples if s not in fpr_samples]
        missing_in_activity = [s for s in fpr_samples if s not in activity_samples]

        if len(missing_in_fpr) > 0 or len(missing_in_activity) > 0:
            st.warning("Sample mismatch between activities and FPR files.")
            if len(missing_in_fpr) > 0:
                st.write("Present in activities but missing in FPR:")
                st.write(missing_in_fpr)
            if len(missing_in_activity) > 0:
                st.write("Present in FPR but missing in activities:")
                st.write(missing_in_activity)

        shared_samples = [s for s in activity_samples if s in fpr_samples]

        if len(shared_samples) == 0:
            st.error("No overlapping samples were found between the activities and FPR files. Make sure both files come from the same KSTAR run.")
            st.stop()

        activity = activity[activity["Sample"].isin(shared_samples)].copy()
        fpr = fpr[fpr["Sample"].isin(shared_samples)].copy()

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
        working = working.iloc[np.argsort(-conf_score)[:topn_conf]]

    file_k_order = list(pd.Index(merged["Kinase"]).drop_duplicates())
    file_s_order = list(pd.Index(merged["Sample"]).drop_duplicates())

    y_order = file_k_order
    x_order = file_s_order

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

    nl = (-np.log10(working["FPR"])).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    bins = [0, 1, 2, 3, np.inf]
    labels = [6, 10, 14, 18]
    idx = np.digitize(nl, bins) - 1
    size_vals = np.array(labels)[np.clip(idx, 0, len(labels) - 1)]
    colors = np.where(working["Significant"] == "Sig", "#d62728", "#d3d3d3")

    fig_dot = go.Figure()

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

# 2) Ranked bar plot
st.divider()
st.subheader("2) Ranked Inferred Kinase Activity Summary")
st.markdown(
    "This bar plot ranks inferred kinase activities so the strongest signals are easier to prioritize. "
    "Unlike the dot plot, which shows every kinase–sample pair, this view summarizes each inferred kinase activity across samples. "
    "By default, activities are ranked by their best false positive rate (FPR), which is usually more informative than raw mean score. "
    "Entries with no nonzero signal under the selected ranking method are listed separately below the plot."
)

rank_metric_col1, rank_metric_col2 = st.columns([1.2, 1])
with rank_metric_col1:
    rank_metric = st.selectbox(
        "Rank inferred kinase activities by",
        ["Best FPR", "Significant Sample Count", "Mean Score", "Median Score"],
        index=0,
    )
with rank_metric_col2:
    ranked_fpr_thr = st.number_input(
        "Significance threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.05,
        step=0.01,
        key="ranked_fpr_thr",
    )

rank_work = merged.copy()
rank_work["Score"] = pd.to_numeric(rank_work["Score"], errors="coerce")
if "FPR" in rank_work.columns:
    rank_work["FPR"] = pd.to_numeric(rank_work["FPR"], errors="coerce")

ranked = (
    rank_work.groupby("Kinase", dropna=False)
    .agg(
        MeanScore=("Score", "mean"),
        MedianScore=("Score", "median"),
        NumSamples=("Sample", "nunique"),
    )
    .reset_index()
)

if "FPR" in rank_work.columns and not rank_work["FPR"].isna().all():
    fpr_summary = (
        rank_work.groupby("Kinase", dropna=False)
        .agg(
            BestFPR=("FPR", "min"),
            MedianFPR=("FPR", "median"),
            SignificantSampleCount=(
                "FPR",
                lambda x: int(np.sum(pd.to_numeric(x, errors="coerce") <= ranked_fpr_thr)),
            ),
        )
        .reset_index()
    )
    ranked = ranked.merge(fpr_summary, on="Kinase", how="left")
else:
    ranked["BestFPR"] = np.nan
    ranked["MedianFPR"] = np.nan
    ranked["SignificantSampleCount"] = 0

ranked["PriorityGroup"] = "Below selected threshold"
zero_signal_entries = pd.DataFrame()

if rank_metric == "Mean Score":
    ranked["PlotValue"] = pd.to_numeric(ranked["MeanScore"], errors="coerce")
    ranked = ranked.sort_values("PlotValue", ascending=False)
    ranked["PriorityGroup"] = np.where(
        ranked["PlotValue"] > 0,
        "Has nonzero mean score",
        "No measurable mean score",
    )
    ranked_plot = ranked[ranked["PlotValue"] > 0].copy()
    zero_signal_entries = ranked[ranked["PlotValue"] <= 0].copy()
    x_col = "Kinase"
    y_col = "PlotValue"
    y_label = "Mean inferred kinase activity score"
    plot_title = f"All inferred kinase activities with nonzero mean score"

elif rank_metric == "Median Score":
    ranked["PlotValue"] = pd.to_numeric(ranked["MedianScore"], errors="coerce")
    ranked = ranked.sort_values("PlotValue", ascending=False)
    ranked["PriorityGroup"] = np.where(
        ranked["PlotValue"] > 0,
        "Has nonzero median score",
        "No measurable median score",
    )
    ranked_plot = ranked[ranked["PlotValue"] > 0].copy()
    zero_signal_entries = ranked[ranked["PlotValue"] <= 0].copy()
    x_col = "Kinase"
    y_col = "PlotValue"
    y_label = "Median inferred kinase activity score"
    plot_title = f"All inferred kinase activities with nonzero median score"

elif rank_metric == "Best FPR":
    ranked_plot = ranked.dropna(subset=["BestFPR"]).copy()
    ranked_plot = ranked_plot[ranked_plot["BestFPR"] > 0].copy()

    if len(ranked_plot) > 0:
        ranked_plot["PlotValue"] = -np.log10(
            ranked_plot["BestFPR"].clip(lower=1e-300)
        )
        ranked_plot["PriorityGroup"] = np.where(
            ranked_plot["BestFPR"] <= ranked_fpr_thr,
            "Meets significance threshold",
            "Does not meet significance threshold",
        )
        ranked_plot = ranked_plot.sort_values("BestFPR", ascending=True)

    zero_signal_entries = ranked[
        ranked["BestFPR"].isna() | (pd.to_numeric(ranked["BestFPR"], errors="coerce") <= 0)
    ].copy()

    x_col = "Kinase"
    y_col = "PlotValue"
    y_label = "-log10(best FPR) for inferred kinase activity"
    plot_title = "All inferred kinase activities with valid best FPR"

else:
    ranked["PlotValue"] = pd.to_numeric(ranked["SignificantSampleCount"], errors="coerce")
    ranked = ranked.sort_values(["PlotValue", "BestFPR"], ascending=[False, True])
    ranked["PriorityGroup"] = np.where(
        ranked["PlotValue"] > 0,
        "At least one significant sample",
        "No significant samples",
    )
    ranked_plot = ranked[ranked["PlotValue"] > 0].copy()
    zero_signal_entries = ranked[ranked["PlotValue"] <= 0].copy()
    x_col = "Kinase"
    y_col = "PlotValue"
    y_label = f"Number of samples with inferred kinase activity FPR ≤ {ranked_fpr_thr:g}"
    plot_title = "All inferred kinase activities with at least one significant sample"

if len(ranked_plot) == 0:
    st.info("No inferred kinase activities had a nonzero plotted value for the selected ranking method.")
else:
    fig_rank = px.bar(
        ranked_plot,
        x=x_col,
        y=y_col,
        color="PriorityGroup",
        color_discrete_map={
            "Meets significance threshold": "#d62728",
            "Does not meet significance threshold": "#bdbdbd",
            "Has nonzero mean score": "#4c78a8",
            "Has nonzero median score": "#4c78a8",
            "At least one significant sample": "#d62728",
            "No significant samples": "#bdbdbd",
        },
        hover_data={
            "MeanScore": ":.4f",
            "MedianScore": ":.4f",
            "BestFPR": ":.4g",
            "MedianFPR": ":.4g",
            "SignificantSampleCount": True,
            "NumSamples": True,
            "PriorityGroup": False,
        },
        title=plot_title,
    )

    if rank_metric == "Best FPR":
        fig_rank.add_hline(
            y=-np.log10(max(ranked_fpr_thr, 1e-300)),
            line_dash="dash",
            annotation_text="Selected significance threshold",
            annotation_position="top right",
        )

    y_max = pd.to_numeric(ranked_plot[y_col], errors="coerce").max()
    if pd.isna(y_max) or y_max <= 0:
        y_max = 1

    fig_rank.update_layout(
        margin=dict(l=40, r=20, t=60, b=140),
        height=600,
        xaxis_tickangle=-45,
        yaxis_title=y_label,
        xaxis_title="Inferred kinase activity target",
        legend_title_text="",
        yaxis_range=[0, y_max * 1.08],
    )

    st.plotly_chart(fig_rank, use_container_width=True)
    fig_download_controls(fig_rank, "kstar_ranked_inferred_kinase_activity_barplot", "ranked_bar_dl")

if len(zero_signal_entries) > 0:
    zero_names = zero_signal_entries["Kinase"].dropna().astype(str).tolist()

    if rank_metric == "Best FPR":
        zero_title = f"Inferred kinase activities with no valid best FPR to plot ({len(zero_names)})"
        zero_text = (
            "These inferred kinase activities did not have a valid positive best FPR value available for plotting "
            "under the current settings."
        )
    elif rank_metric == "Significant Sample Count":
        zero_title = f"Inferred kinase activities with no significant samples ({len(zero_names)})"
        zero_text = (
            "These inferred kinase activities had zero samples meeting the selected significance threshold."
        )
    elif rank_metric == "Mean Score":
        zero_title = f"Inferred kinase activities with no nonzero mean score ({len(zero_names)})"
        zero_text = (
            "These inferred kinase activities had mean scores of zero or missing values under the current settings."
        )
    else:
        zero_title = f"Inferred kinase activities with no nonzero median score ({len(zero_names)})"
        zero_text = (
            "These inferred kinase activities had median scores of zero or missing values under the current settings."
        )

    with st.expander(zero_title, expanded=False):
        st.markdown(zero_text)
        st.markdown(", ".join(zero_names))

# 3) Activity Heatmap
st.divider()
st.subheader("3) Activity Heatmap")
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

# 4) Data preview
st.divider()
st.subheader("4) Data Preview")
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

# 5) Kinase detail
st.divider()
st.subheader("5) Kinase Detail")
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
