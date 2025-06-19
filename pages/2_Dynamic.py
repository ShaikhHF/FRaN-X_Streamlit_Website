import streamlit as st
from sidebar import render_sidebar
from load_annotations import load_article, load_labels, load_file_names
from Home import reformat_text_html_with_tooltips

ROLE_COLORS = {
    "Protagonist": "#a1f4a1",
    "Antagonist": "#f4a1a1",
    "Innocent": "#a1c9f4",
}

st.set_page_config(page_title="Compare", page_icon="📈", layout="wide")

# Sidebar
article, labels, use_example, threshold, role_filter = render_sidebar(False)

# Title Row with Dynamic Column Buttons
title_col, spacer, add_col, remove_col = st.columns([3, 5, 1, 1])
with title_col:
    st.markdown("# Compare Articles")
with add_col:
    if st.button("➕", use_container_width=True):
        st.session_state.column_count = min(st.session_state.get("column_count", 1) + 1, 4)
with remove_col:
    if st.button("➖", use_container_width=True):
        st.session_state.column_count = max(st.session_state.get("column_count", 1) - 1, 1)

st.write("Select one or more articles to compare their role distributions and contents side-by-side.")

# Initialize column count
column_count = st.session_state.get("column_count", 2)

# Prepare file options
use_example = st.session_state.get("use_example", False)
threshold = st.session_state.get("threshold", 0.5)
article_folder = 'chunk_data' if use_example else 'user_articles'
label_folder = 'split_data' if use_example else 'user_labels'

file_names = [f for f in load_file_names(article_folder) if f and not f.startswith('.')]
file_options = ["Select a file"] + file_names

# Render dynamic columns
columns = st.columns(column_count)
for i, col in enumerate(columns):
    with col:
        st.markdown(f"#### Article {i + 1}")
        selected_file = st.selectbox(f"Choose a file ({i + 1})", file_options, key=f"file_{i}")
        
        if selected_file != "Select a file":
            article_text = load_article(f"{article_folder}/{selected_file}")
            st.text_area("Content", article_text, height=300, key=f"text_area_{i}")
            
            labels = load_labels(label_folder, selected_file, threshold)
            html = reformat_text_html_with_tooltips(article_text, labels)
            st.components.v1.html(html, height=600)
