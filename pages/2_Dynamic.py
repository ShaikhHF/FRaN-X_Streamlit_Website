import streamlit as st
import pandas as pd
import altair as alt
from sidebar import render_sidebar
from load_annotations import load_article, load_labels, load_file_names
from render_text import reformat_text_html_with_tooltips, predict_entity_framing

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

# Store role distributions for each article
distribution_data = []

# Render dynamic columns
columns = st.columns(column_count)
for i, col in enumerate(columns):
    with col:
        st.markdown(f"#### Article {i + 1}")
        selected_file = st.selectbox(f"Choose a file ({i + 1})", file_options, key=f"file_{i}")
        
        if selected_file != "Select a file":
            article_text = load_article(f"{article_folder}/{selected_file}")
            labels = load_labels(label_folder, selected_file, threshold)
            

            html = reformat_text_html_with_tooltips(article_text, labels)
            line_count = article_text.count("\n") + 1
            estimated_height = min(1500, line_count * 100)
            st.components.v1.html(html, height=estimated_height)

            # Role distribution collection
            df_f = predict_entity_framing(article_text, labels, threshold)

            df_f = df_f[df_f['main_role'].isin(role_filter)]
            if not df_f.empty:
                counts = df_f['main_role'].value_counts().reset_index()
                counts.columns = ['main_role', 'count']
                counts['article'] = selected_file
                distribution_data.append(counts)

# Compare Distributions Across Articles
if distribution_data:
    st.markdown("## Main Role Distribution Comparison")

    # Combine all role counts
    combined_df = pd.concat(distribution_data)

    # Compute percentages for 100% stacked chart
    total_per_article = combined_df.groupby('article')['count'].transform('sum')
    combined_df['percentage'] = combined_df['count'] / total_per_article * 100

    # Sort roles and articles for consistent color/domain
    domain_list = list(ROLE_COLORS.keys())
    color_list = [ROLE_COLORS.get(role, "#cccccc") for role in domain_list]

    # Altair stacked horizontal bar chart
    chart = alt.Chart(combined_df).mark_bar().encode(
        y=alt.Y('article:N', title="Article", sort=None),
        x=alt.X('percentage:Q', stack='normalize', title='Role Composition (%)'),
        color=alt.Color('main_role:N', scale=alt.Scale(domain=domain_list, range=color_list), legend=alt.Legend(title="Main Role")),
        tooltip=['article', 'main_role', 'count', alt.Tooltip('percentage:Q', format=".1f")]
    ).properties(
        width=700,
        height=200 * len(combined_df['article'].unique())
    )

    st.altair_chart(chart, use_container_width=True)

