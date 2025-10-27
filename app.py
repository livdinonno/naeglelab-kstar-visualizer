import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="KSTAR Visualizer", layout="wide")
st.title("KSTAR Kinase Activity Visualizer")

st.markdown("""
Upload the two final KSTAR outputs:
- **_mann_whitney_fpr.tsv** (FPR values)
- **_mann_whitney_activities.tsv** (activity scores)

Or check “Use repository files” if you’ve committed them to this repo.
""")

use_repo = st.checkbox("Use repository files in this repo", value=False)

fpr_df = None
act_df = None

if use_repo:
    default_fpr = "TKCC_Y_mann_whitney_fpr.tsv"
    default_act = "TKCC_Y_mann_whitney_activities.tsv"
    try:
        fpr_df = pd.read_csv(default_fpr, sep="\t")
        act_df = pd.read_csv(default_act, sep="\t")
        st.success("Loaded repository files.")
    except Exception as e:
        st.error(f"Could not load repo files: {e}")
else:
    col1, col2 = st.columns(2)
    with col1:
        fpr_up = st.file_uploader("Upload *mann_whitney_fpr.tsv*", type=["tsv"])
    with col2:
        act_up = st.file_uploader("Upload *mann_whitney_activities.tsv*", type=["tsv"])
    if fpr_up is not None and act_up is not None:
        try:
            fpr_df = pd.read_csv(fpr_up, sep="\t")
            act_df = pd.read_csv(act_up, sep="\t")
            st.success("Uploaded files loaded.")
        except Exception as e:
            st.error(f"Error reading uploaded files: {e}")

if fpr_df is None or act_df is None:
    st.info("Waiting for files… (upload both, or enable 'Use repository files').")
    st.stop()

if "KSTAR_KINASE" not in fpr_df.columns:
    st.error("FPR file is missing 'KSTAR_KINASE' column.")
    st.stop()
if "KSTAR_KINASE" not in act_df.columns:
    st.error("Activities file is missing 'KSTAR_KINASE' column.")
    st.stop()

shared_cols = ["KSTAR_KINASE"] + [c for c in fpr_df.columns if c in act_df.columns and c != "KSTAR_KINASE"]
fpr_df = fpr_df[shared_cols]
act_df = act_df[shared_cols]

with st.expander("Preview: FPR (first 10 rows)"):
    st.dataframe(fpr_df.head(10))
with st.expander("Preview: Activities (first 10 rows)"):
    st.dataframe(act_df.head(10))

st.subheader("FPR Heatmap")
plt.figure(figsize=(11, 8))
sns.heatmap(
    fpr_df.set_index("KSTAR_KINASE"),
    cmap="viridis",
    cbar_kws={"label": "FPR"},
    vmin=0.0, vmax=1.0
)
plt.xlabel("Samples")
plt.ylabel("Kinases")
st.pyplot(plt.gcf())
plt.close()


