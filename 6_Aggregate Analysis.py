import streamlit as st
import os
from sidebar import render_sidebar, load_file_names
from load_annotations import load_article, load_labels
from render_text import reformat_text_html_with_tooltips

st.set_page_config(page_title="FRaN-X", initial_sidebar_state='expanded', layout="wide")
st.title("Aggregate Visualization")

article, labels, user_folder, threshold, role_filter, hide_repeat = render_sidebar(True, True, False, False)

folder_path = 'chunk_data' if user_folder == None else 'user_articles'


if user_folder != None:
    folder_path = os.path.join(folder_path, user_folder)
files = st.sidebar.multiselect(
    "File(s)",
    options=list(load_file_names(folder_path))
)



st.markdown("---")
st.markdown("*UGRIP 2025 FRaN-X Team* ")