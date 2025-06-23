import streamlit as st
import pandas as pd
import altair as alt
import colorsys
from streamlit_echarts import st_echarts
import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network
import tempfile
import os
import streamlit.components.v1 as components
from collections import Counter
import itertools
from sidebar import render_sidebar
from load_annotations import load_article, load_labels, load_file_names
from render_text import reformat_text_html_with_tooltips, predict_entity_framing

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
        rgb_to_hex(colorsys.hls_to_rgb(h, max(0, min(1, l + i * 0.04)), s))
        for i in range(n)
    ]


st.set_page_config(page_title="FRaN-X", layout="wide")

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
            "textStyle": {"color": "#333"}
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

    # Network Graphs
    st.markdown("## Network Graph  ")
    st.write("Explore the relationship between entities and roles across documents to see how groups of entity+role pairs can be used to identify potential narratives.")

    network_rows = []
    for i in range(column_count):
        selected_file = st.session_state.get(f"file_{i}")
        if selected_file and selected_file != "Select a file":
            article_text = load_article(f"{article_folder}/{selected_file}")
            labels = load_labels(label_folder, selected_file, threshold)
            df_network = predict_entity_framing(article_text, labels, threshold)
            df_network = df_network[df_network['main_role'].isin(role_filter)]
            df_network = df_network.explode("fine_roles")
            df_network['fine_roles'] = df_network['fine_roles'].str.strip().str.title()
            df_network = df_network[df_network['fine_roles'].notnull() & (df_network['fine_roles'] != '')]
            df_network['article'] = selected_file
            network_rows.append(df_network)

    if network_rows:
        graph_df = pd.concat(network_rows)

        # Create node labels
        graph_df["node_label"] = graph_df["entity"] + " (" + graph_df["fine_roles"] + ")"

        # Count mentions for node sizes
        node_sizes = graph_df["node_label"].value_counts().to_dict()

        # Assign colors based on main_role
        main_roles = graph_df["main_role"].unique()

        co_occurrence = Counter()
        for _, doc_df in graph_df.groupby("article"):
            nodes = doc_df["node_label"].unique()
            pairs = itertools.combinations(sorted(nodes), 2)
            co_occurrence.update(pairs)

        # Create graph: nodes = entity+role, edges = co-occurrence in the same document
        G = nx.Graph()
        for node, row in graph_df.groupby("node_label").first().iterrows():
            G.add_node(node, role=row["main_role"])

        # Add edges with weight = co-occurrence count
        for (node1, node2), weight in co_occurrence.items():
            G.add_edge(node1, node2, weight=weight)

        max_weight = max(nx.get_edge_attributes(G, "weight").values(), default=1)
        edge_widths = [G[u][v]["weight"] * 2 for u, v in G.edges()]  # scale up for visibility

        # Draw graph
        node_colors = [
            ROLE_COLORS.get(G.nodes[node]['role'], "#cccccc")
            for node in G.nodes()
        ]

        plt.figure(figsize=(14, 14))
        pos = nx.spring_layout(G, k=2, iterations=100, seed=42)

        # Get x and y coordinates of all nodes
        x_vals, y_vals = zip(*pos.values())

        # Set axis limits with padding
        plt.xlim(min(x_vals) - 0.3, max(x_vals) + 0.3)
        plt.ylim(min(y_vals) - 0.3, max(y_vals) + 0.3)


        node_sizes = [node_sizes.get(node, 1) * 900 for node in G.nodes()]
        node_colors = [
            ROLE_COLORS.get(G.nodes[node]['role'], "#cccccc") for node in G.nodes()
        ]

        nx.draw_networkx_edges(G, pos, alpha=0.3, width=edge_widths)
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9)
        nx.draw_networkx_labels(G, pos, font_size=10)

        #plt.title("Entity+Role Narrative Graph with Document Co-Occurrence Weights", fontsize=16)
        plt.axis("off")
        plt.tight_layout()
        st.pyplot(plt.gcf())

    # --- Entity → Fine Role Network Graph ---
    st.markdown("## Interactive Graph ")
    st.write("Interact with the connection between the entity,role nodes present across multiple documents. The edges represent a document where the two nodes are both present. ")

    if not graph_df.empty:
        import networkx as nx
        from pyvis.network import Network
        import streamlit.components.v1 as components
        import tempfile
        import os

        # Build the graph
        G = nx.Graph()
        for _, row in graph_df.iterrows():
            node_id = f"{row['entity']} → {row['fine_roles']}"
            role = row['main_role']
            color = ROLE_COLORS.get(role, "#cccccc")

            if G.has_node(node_id):
                G.nodes[node_id]['size'] += 1
            else:
                G.add_node(node_id, label=node_id, color=color, size=15)

        for _, group in graph_df.groupby('article'):
            entities = group.apply(lambda r: f"{r['entity']} → {r['fine_roles']}", axis=1).unique()
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    e1, e2 = entities[i], entities[j]
                    if G.has_edge(e1, e2):
                        G[e1][e2]['weight'] += 1
                    else:
                        G.add_edge(e1, e2, weight=1)

        # Create and render the network
        net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
        net.from_nx(G)
        net.repulsion(node_distance=150, spring_length=200)
        net.show_buttons(filter_=["physics"])

        # Use temporary file properly
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
            net.save_graph(tmp_file.name)
            tmp_path = tmp_file.name

        with open(tmp_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        components.html(html_content, height=1000, scrolling=True)

        # Optional cleanup
        os.remove(tmp_path)