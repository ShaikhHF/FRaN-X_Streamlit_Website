import streamlit as st
import time
import numpy as np
from sidebar import render_sidebar

ROLE_COLORS = {
    "Protagonist": "#a1f4a1",
    "Antagonist":  "#f4a1a1",
    "Innocent":    "#a1c9f4",
}

st.set_page_config(page_title="Compare", page_icon="📈")

article, labels, use_example, threshold, role_filter = render_sidebar()

st.markdown("# Plotting Demo")

st.write(
    """This demo illustrates a combination of plotting and animation with
Streamlit. We're generating a bunch of random numbers in a loop for around
5 seconds. Enjoy!"""
)