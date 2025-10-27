import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="KSTAR Visualizer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# for light theme (white background)
st.markdown(
    """
    <style>
        :root {
            color-scheme: light;
        }
        body {
            background-color: white !important;
            color: black !important;
        }
        [data-testid="stAppViewContainer"] {
            background-color: white !important;
        }
        [data-testid="stHeader"] {
            background: white !important;
        }
        [data-testid="stSidebar"] {
            background-color: #f9f9f9 !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

def home_page():
    st.title("Publication on Phosphotyrosine Profiling")
    st.markdown("Welcome to the KSTAR Visualizer. This site allows you to visualize your data after processing it through the KSTAR algorithm. It also links relevant publications and tutorials.")

if __name__ == "__main__":
    home_page()

