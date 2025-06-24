import streamlit as st
from sidebar import render_sidebar, load_file_names
from load_annotations import load_article, load_labels
from render_text import reformat_text_html_with_tooltips

ROLE_COLORS = {
   "Protagonist": "#a1f4a1",
   "Antagonist":  "#f4a1a1",
   "Innocent":    "#a1c9f4",
}
st.set_page_config(page_title="FRaN-X", initial_sidebar_state='expanded', layout="wide")
st.title("Search")

article, labels, use_example, threshold, role_filter = render_sidebar(False)

folder_path = 'chunk_data' if use_example else 'user_articles'
files = st.sidebar.multiselect(
    "Select Files",
    options=list(load_file_names(folder_path))
)

word = st.text_input("Search for: ")

for f in files:
    article = load_article(f'{folder_path}/{f}')
    if word in article:
        labels = load_labels(
            'split_data' if use_example else 'user_articles',
            f,
            threshold
        )

        st.write(f"#### {f} \n", height=100, scrolling=True)

        with st.expander("Article", expanded=False):
            html = reformat_text_html_with_tooltips(article, labels, word)
            st.components.v1.html(html, height=600, scrolling = True)     



st.markdown("---")
st.markdown("*UGRIP 2025 FRaN-X Team* ")