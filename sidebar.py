# sidebar.py
import streamlit as st
from load_annotations import load_article, load_labels, load_file_names

ROLE_COLORS = {
    "Protagonist": "#a1f4a1",
    "Antagonist":  "#f4a1a1",
    "Innocent":    "#a1c9f4",
}



def render_sidebar():
    st.sidebar.header("Settings")

    if "use_example" not in st.session_state:
        st.session_state.use_example = False
    use_example = st.sidebar.checkbox("Use example article for illustration", value=st.session_state.use_example)
    st.session_state.use_example = use_example

    folder_path = 'chunk_data' if use_example else 'user_articles'
    file_names = load_file_names(folder_path)

    if "article_name" not in st.session_state and file_names:
        st.session_state.article_name = file_names[0]
    article_name = st.sidebar.selectbox("Choose a file", file_names, index=file_names.index(st.session_state.article_name) if st.session_state.article_name in file_names else 0)
    st.session_state.article_name = article_name

    if "threshold" not in st.session_state:
        st.session_state.threshold = 0.5
    threshold = st.sidebar.slider("Narrative confidence threshold", 0.0, 1.0, st.session_state.threshold, 0.01)
    st.session_state.threshold = threshold

    if "role_filter" not in st.session_state:
        st.session_state.role_filter = list(ROLE_COLORS.keys())
    role_filter = st.sidebar.multiselect(
        "Filter roles",
        options=list(ROLE_COLORS.keys()),
        default=st.session_state.role_filter
    )

    article = ''
    labels = []
    if isinstance(article_name, str):
        article = load_article(f'{folder_path}/{article_name}')
        labels = load_labels('split_data' if use_example else 'user_articles', article_name)

    return article, labels, use_example, threshold, role_filter
