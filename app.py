import streamlit as st
import pandas as pd

# GitHub tutorial link
TUTORIAL_URL = "https://docs.github.com/en/get-started/start-your-journey/hello-world" 

st.set_page_config(
    page_title="KSTAR Visualizer",
    layout="wide",
    initial_sidebar_state="collapsed"  # sidebar starts closed
)

# background color
st.markdown(
    """
    <style>
    .stApp { background-color: #FFFFFF; }
    </style>
    """,
    unsafe_allow_html=True
)

# home page
def home_page():
    st.title("Welcome to the KSTAR Visualizer")
    st.markdown(
        "This site allows you to visualize your data after processing it through the KSTAR algorithm. "
        "It also provides quick access to relevant publications and a setup tutorial."
    )

    st.link_button("Open GitHub Tutorial", TUTORIAL_URL)
    st.caption("Click the button above to view the step-by-step tutorial on GitHub.")
    st.divider()

    # main-page expander for related publications
    with st.expander("Related Publications", expanded=False):
        publications = [
            {
                "title": "Phosphotyrosine Profiling Reveals New Signaling Networks (PMC Article)",
                "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4974343/"
            },
        ]
        for pub in publications:
            st.markdown(f"- [{pub['title']}]({pub['url']})")

    st.divider()
    st.markdown("### Ready to explore your data?")
    st.markdown("Upload your KSTAR output files to begin visualizing your results.")
    if st.button("Launch Visualizer"):
        st.session_state.page = "visualizer"

# visualizer page
def visualizer_page():
    st.title("KSTAR Data Visualizer")
    st.caption("Upload the two final KSTAR outputs: mann_whitney_fpr.tsv and mann_whitney_activities.tsv")

    col1, col2 = st.columns(2)
    with col1:
        fpr_file = st.file_uploader("Upload FPR file (mann_whitney_fpr.tsv)", type=["tsv"], key="fpr")
    with col2:
        act_file = st.file_uploader("Upload Activities file (mann_whitney_activities.tsv)", type=["tsv"], key="act")

    if fpr_file and act_file:
        try:
            fpr_df = pd.read_csv(fpr_file, sep="\t")
            act_df = pd.read_csv(act_file, sep="\t")
            st.success("Files loaded successfully.")

            st.subheader("Preview")
            t1, t2 = st.tabs(["FPR (first 20 rows)", "Activities (first 20 rows)"])
            with t1:
                st.dataframe(fpr_df.head(20))
            with t2:
                st.dataframe(act_df.head(20))
        except Exception as e:
            st.error(f"Could not read one or both files: {e}")
    else:
        st.info("Please upload both .tsv files to begin.")

    st.divider()
    if st.button("Back to Home"):
        st.session_state.page = "home"

# sidebar
# def render_sidebar():
 #   st.sidebar.title("Related Publications")
  #  publications = [
   #     {
    #        "title": "Phosphotyrosine Profiling Reveals New Signaling Networks (PMC Article)",
     #       "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4974343/"
      #  },
   # ]
   # for pub in publications:
    #    st.sidebar.markdown(f"- [{pub['title']}]({pub['url']})")

# main router
#if __name__ == "__main__":
#    render_sidebar()

    # page switching logic
  #  if "page" not in st.session_state:
  #      st.session_state.page = "home"

  #  if st.session_state.page == "home":
   #     home_page()
  #  elif st.session_state.page == "visualizer":
   #     visualizer_page()




