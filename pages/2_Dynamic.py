import streamlit as st
import pandas as pd
import altair as alt
from sidebar import render_sidebar
from load_annotations import load_article, load_labels, load_file_names
from render_text import reformat_text_html_with_tooltips, predict_entity_framing
import colorsys
from streamlit_echarts import st_echarts

ROLE_COLORS = {
    "Protagonist": "#a1f4a1",
    "Antagonist": "#f4a1a1",
    "Innocent": "#a1c9f4",
}

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4))

def rgb_to_hex(rgb_tuple):
    return "#{:02x}{:02x}{:02x}".format(
        int(rgb_tuple[0]*255), int(rgb_tuple[1]*255), int(rgb_tuple[2]*255)
    )

def generate_shades(base_hex, n):
    base_rgb = hex_to_rgb(base_hex)
    h, l, s = colorsys.rgb_to_hls(*base_rgb)
    return [
        rgb_to_hex(colorsys.hls_to_rgb(h, max(0, min(1, l + i * 0.05)), s))
        for i in range(n)
    ]


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
            st.components.v1.html(html, height=800, scrolling = True)

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

# --- Cumulative Pie Chart by Main Role → Fine-Grained Role ---
    st.markdown("## Cumulative Role Breakdown")

    # Reload detailed entity data for pie chart
    all_detailed_data = []
    for i, col in enumerate(columns):
        selected_file = st.session_state.get(f"file_{i}")
        if selected_file and selected_file != "Select a file":
            article_text = load_article(f"{article_folder}/{selected_file}")
            labels = load_labels(label_folder, selected_file, threshold)
            df_detail = predict_entity_framing(article_text, labels, threshold)
            df_detail = df_detail[df_detail['main_role'].isin(role_filter)]
            all_detailed_data.append(df_detail)

    if all_detailed_data:
        cumulative_df = pd.concat(all_detailed_data)

        # Explode fine_roles
        cumulative_df = cumulative_df.explode("fine_roles")
        cumulative_df['fine_roles'] = cumulative_df['fine_roles'].str.strip().str.title()
        cumulative_df = cumulative_df[cumulative_df['fine_roles'].notnull() & (cumulative_df['fine_roles'] != '')]

        # Group by both main_role and fine_role
        role_fine_counts = cumulative_df.groupby(['main_role', 'fine_roles']).size().reset_index(name='count')

        # Manually assign color shades per role
        fine_role_colors = {}
        for role in role_fine_counts['main_role'].unique():
            fine_roles = role_fine_counts[role_fine_counts['main_role'] == role]['fine_roles'].unique()
            base_color = ROLE_COLORS.get(role, "#cccccc")
            shades = generate_shades(base_color, len(fine_roles))
            for fine, shade in zip(fine_roles, shades):
                fine_role_colors[fine] = shade

        color_scale = alt.Scale(domain=list(fine_role_colors.keys()), range=list(fine_role_colors.values()))

        # Compute percentages for labeling
        role_fine_counts['percentage'] = (
            role_fine_counts.groupby('main_role')['count']
            .transform(lambda x: x / x.sum() * 100)
        )

        # Base chart with shared encodings
        base = alt.Chart(role_fine_counts).encode(
            theta=alt.Theta(field='count', type='quantitative'),
            color=alt.Color('fine_roles:N', scale=color_scale, legend=alt.Legend(title='Fine Role')),
            tooltip=['main_role', 'fine_roles', 'count', alt.Tooltip('percentage:Q', format=".1f")]
        )

        # Arc chart
        arc = base.mark_arc(innerRadius=30)


        # Combine and facet
        pie_chart = (arc).facet(
            column=alt.Column('main_role:N', title=None, header=alt.Header(labelAngle=0))
        ).properties(
            title="Fine-Grained Roles per Main Role"
        )

        st.altair_chart(pie_chart, use_container_width=True)


    echarts_data = [
        {"value": int(row["count"]), "name": f"{row['main_role']} → {row['fine_roles']}"}
        for _, row in role_fine_counts.iterrows()
    ]

    # ECharts options
    options = {
        "title": {"text": "Fine-Grained Role Distribution", "left": "center"},
        "tooltip": {"trigger": "item"},
        "legend": {
            "orient": "vertical",
            "left": "left",
            "textStyle": {"color": "#333"}  # optional: adjusts legend label color
        },
        "series": [
            {
                "name": "Role Breakdown",
                "type": "pie",
                "radius": ["30%", "70%"],  # donut-style for clarity
                "label": {
                    "show": True,
                    "formatter": "{b}\n({d}%)",
                    "fontSize": 12,
                    "color": "#000000",
                    "fontWeight": "bold",
                    "overflow": "truncate"
                },
                "labelLine": {"show": False},
                "data": echarts_data,
                "emphasis": {
                    "itemStyle": {
                        "shadowBlur": 10,
                        "shadowOffsetX": 0,
                        "shadowColor": "rgba(0, 0, 0, 0.5)"
                    }
                },
            }
        ],
    }

    st_echarts(options=options, height="600px")