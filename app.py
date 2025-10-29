import streamlit as st
import pandas as pd

# GitHub tutorial link
TUTORIAL_URL = "https://docs.github.com/en/get-started/start-your-journey/hello-world" 

st.set_page_config(
    page_title="KSTAR Visualizer",
    layout="wide",
    initial_sidebar_state="expanded"
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

def home_page():
    st.title("Welcome to the KSTAR Visualizer")
    st.markdown(
        "This site allows you to visualize your data after processing it through the KSTAR algorithm."
        "It also provides quick access to relevant publications and a setup tutorial."
    )

    st.link_button("Open GitHub Tutorial", TUTORIAL_URL)
    st.caption("Click the button above to view the step-by-step tutorial on GitHub.")

def render_sidebar():
    st.sidebar.title("Related Publications")

    publications = [
        {
            "title": "Phosphotyrosine Profiling Reveals New Signaling Networks (PMC Article)",
            "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4974343/"
        },
        # Add more publication entries here if needed later
    ]

    for pub in publications:
        st.sidebar.markdown(f"- [{pub['title']}]({pub['url']})")

if __name__ == "__main__":
    render_sidebar()
    home_page()


