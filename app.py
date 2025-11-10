import streamlit as st
import pandas as pd

# GitHub tutorial link
TUTORIAL_URL = "https://docs.github.com/en/get-started/start-your-journey/hello-world" 
# Run KSTAR data link
KSTAR_URL = "https://naeglelab-test-proteome-scout-3.pods.uvarc.io/kstar/"

st.set_page_config(
    page_title="KSTAR Results Plotter",
    layout="wide",
    # initial_sidebar_state="collapsed"  # sidebar starts closed
)

# home page
def home_page():
    st.markdown(
        """
        <div style='background-color:#f5f5f5; padding: 2rem; border-radius: 8px; text-align: center;'>
            <h1>Welcome to the KSTAR Results Plotter</h1>
            <p>This site allows you to visualize your data after processing it through the KSTAR algorithm.
            It also provides quick access to a setup tutorial and relevant publications.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()
    st.markdown("### Ready to explore your data?")
    st.markdown("Upload your KSTAR output files to begin visualizing your results.")
    st.markdown(
        f"""
        <a href="{KSTAR_URL}" target="_blank"
        style="
            display: inline-block;
            background-color: #d8f3dc;
            color: #000000;
            font-weight: 600;
            padding: 0.75rem 1.5rem;
            border-radius: 20px;
            text-decoration: none;
            border: 1px solid #b7e4c7;
        ">
        Launch Results Plotter
        </a>
        """,
        unsafe_allow_html=True
    )
    
    st.divider()
    st.markdown(
        f"""
        <div style='background-color:#f5f5f5; padding: 2rem; border-radius: 8px;'>
            <p style='font-size:0.9rem; color:#555555;'>
                Click the button below to view the step-by-step tutorial on GitHub.
            </p>
            <a href="{TUTORIAL_URL}" target="_blank"
               style="
                    display: inline-block;
                    background-color: #ffffff;
                    color: #000000;
                    font-weight: 500;
                    padding: 0.4rem 0.9rem;
                    border-radius: 6px;
                    text-decoration: none;
                    border: 1px solid #e0e0e0;
               ">
               Open GitHub Tutorial
            </a>

            <p style='font-size:0.9rem; color:#555555; margin-top:1.5rem;'>
                See key publications related to KSTAR methodology and applications.
            </p>

            <details style="background-color:#f5f5f5; padding:0.5rem 0; cursor:pointer;">
                <summary style="font-weight:600;">Related Publications</summary>
                <ul>
                    <li>
                        <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC4974343/" target="_blank">
                            Phosphotyrosine Profiling Reveals New Signaling Networks (PMC Article)
                        </a>
                    </li>
                </ul>
            </details>
        </div>
        """,
        unsafe_allow_html=True
    )


# run home page
home_page()



