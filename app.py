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
    .stApp {
        background-color: #FFFFFF; /* White color */
    }
    </style>
    """,
    unsafe_allow_html=True
)

def home_page():
    st.title("Welcome to the KSTAR Visualizer")
    st.markdown("This site allows you to visualize your data after processing it through the KSTAR algorithm. It also links relevant publications and tutorials.")

if __name__ == "__main__":
    home_page()

