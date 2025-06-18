import streamlit as st
import time
import numpy as np
from load_annotations import load_article, load_labels, load_file_names

ROLE_COLORS = {
    "Protagonist": "#a1f4a1",
    "Antagonist":  "#f4a1a1",
    "Innocent":    "#a1c9f4",
}

st.set_page_config(page_title="Compare", page_icon="📈")

# Sidebar controls
st.sidebar.header("Settings")
use_example = st.sidebar.checkbox("Use example article for illustration")
if use_example:
    folder_path = 'chunk_data'
    file_names = load_file_names(folder_path)
else:
    folder_path = 'user_articles'
    file_names = load_file_names(folder_path)

article_name = st.sidebar.selectbox("Choose a file", file_names)
if not isinstance(article_name, str):
    article = ''
else:
    article = load_article(folder_path +'/'+ article_name)
    if use_example:
        labels = load_labels('split_data', article_name) 
    else:
        labels = load_labels('user_articles', article_name)


threshold    = st.sidebar.slider("Narrative confidence threshold", 0.0, 1.0, 0.5, 0.01)
role_filter  = st.sidebar.multiselect("Filter roles", list(ROLE_COLORS.keys()), default=list(ROLE_COLORS.keys()))

st.markdown("# Plotting Demo")

st.write(
    """This demo illustrates a combination of plotting and animation with
Streamlit. We're generating a bunch of random numbers in a loop for around
5 seconds. Enjoy!"""
)