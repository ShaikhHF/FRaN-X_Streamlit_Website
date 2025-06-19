import streamlit as st
from load_annotations import load_article, load_labels, load_file_names

ROLE_COLORS = {
    "Protagonist": "#a1f4a1",
    "Antagonist":  "#f4a1a1",
    "Innocent":    "#a1c9f4",
}

def render_sidebar():
    st.sidebar.header("Settings")

    # State initialization
    if "use_example" not in st.session_state:
        st.session_state.use_example = False
    if "threshold" not in st.session_state:
        st.session_state.threshold = 0.5
    if "role_filter" not in st.session_state:
        st.session_state.role_filter = list(ROLE_COLORS.keys())

    # Sidebar widgets
    use_example = st.sidebar.checkbox("Use example article for illustration", value=st.session_state.use_example)
    if use_example != st.session_state.use_example:
        st.session_state.use_example = use_example
        st.rerun()


    threshold = st.sidebar.slider("Narrative confidence threshold", 0.0, 1.0, st.session_state.threshold, 0.01)
    st.session_state.threshold = threshold

    role_filter = st.sidebar.multiselect(
        "Filter roles",
        options=list(ROLE_COLORS.keys()),
        default=st.session_state.role_filter
    )
    st.session_state.role_filter = role_filter

    # Load file names
    folder_path = 'chunk_data' if use_example else 'user_articles'
    file_names = load_file_names(folder_path)

    valid_files = [f for f in file_names if f and not f.startswith('.')]

    if (
        "article_name" not in st.session_state
        or st.session_state.article_name not in valid_files
    ):
        st.session_state.article_name = "Select a file"


    article = ''
    labels = []
    file_list = list(file_names)

    # Add a friendly prompt at the top
    file_options = ["Select a file"] + file_list

    if file_names:
        selected_index = file_options.index(st.session_state.article_name) if st.session_state.article_name in file_options else 0

        st.session_state.article_name = st.sidebar.selectbox(
            "Choose a file",
            file_options,
            index=selected_index
        )

        if st.session_state.article_name != "Select a file":
            article = load_article(f'{folder_path}/{st.session_state.article_name}')
            labels = load_labels(
                'split_data' if use_example else 'user_articles',
                st.session_state.article_name,
                threshold
            )
    else:
        st.sidebar.warning("⚠️ No files found in the selected folder.")

    return article, labels, use_example, threshold, role_filter
