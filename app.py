import streamlit as st
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:
    pass

st.set_page_config(page_title="KSTAR Visualizer", layout="wide", initial_sidebar_state="expanded")

def home_page():
    st.title("Publication on Phosphotyrosine Profiling")
    st.markdown("Welcome to the KSTAR Visualizer. This site allows you to visualize your data after processing it through the KSTAR algorithm. It also links relevant publications and tutorials.")

if __name__ == "__main__":
    home_page()
