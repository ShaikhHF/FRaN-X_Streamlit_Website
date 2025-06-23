import streamlit as st
import pandas as pd
import altair as alt
import requests
from bs4 import BeautifulSoup
import re
from sidebar import render_sidebar
from render_text import reformat_text_html_with_tooltips, predict_entity_framing, format_sentence_with_spans
import os
from streamlit.components.v1 import html as st_html

ROLE_COLORS = {
   "Protagonist": "#a1f4a1",
   "Antagonist":  "#f4a1a1",
   "Innocent":    "#a1c9f4",
}


# Narrative classification
def predict_narrative_classification(text, threshold=0.0):
    data = [
        {"narrative": "Western Interference", "confidence": 0.88},
        {"narrative": "Peace Sabotage",      "confidence": 0.85},
        {"narrative": "Proxy War",           "confidence": 0.80},
    ]
    df = pd.DataFrame(data)
    return df[df.confidence >= threshold]

# Free-form narrative extraction
def extract_narrative(text):
    return "Kremlin claims British pressure and additional Russian demands thwarted a potential Ukraine peace deal."

def escape_entity(entity):
    return re.sub(r'([.^$*+?{}\[\]\\|()])', r'\\\1', entity)


# --- Streamlit App ---

st.set_page_config(page_title="FRaN-X", layout="wide")
st.title("FRaN-X: Entity Framing & Narrative Analysis")


# Article input
st.header("1. Article Input")

article, labels, use_example, threshold, role_filter = render_sidebar()

if use_example:
    st.text_area("Example Article", article, height=300)
else:
    mode = st.radio("Input mode", ["Paste Text","URL"])
    if mode == "Paste Text":
        article = st.text_area("Article text", height=200)
        os.makedirs("user_articles", exist_ok=True)
        filename_input = st.text_input("Filename (without extension)")

        # Save button
        if st.button("Save Article"):
            if article.strip() and filename_input.strip():
                # Clean filename and enforce .txt extension
                safe_filename = filename_input.strip().replace(" ", "_") + ".txt"
                filepath = os.path.join('user_articles', safe_filename)

                # Check for duplicate filenames
                if os.path.exists(filepath):
                    st.error(f"A file named '{safe_filename}' already exists. Please choose a different name.")
                else:
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(article)
                    st.success(f"Article saved as {safe_filename}. Please wait while your entity framing is being calculated")
                    st.rerun()
            else:
                st.warning("Both article text and filename must be provided.")

    else:
        url = st.text_input("Article URL")
        article = ""
        if url:
            resp = requests.get(url)
            soup = BeautifulSoup(resp.content, 'html.parser')
            article = '\n'.join(p.get_text() for p in soup.find_all('p'))

if article and labels:
    show_annot   = st.checkbox("Show annotated article view", True)
    df_f = predict_entity_framing(article, labels, threshold)

    # 2. Annotated article view
    if show_annot:
        st.header("2. Annotated Article")
        html = reformat_text_html_with_tooltips(article, labels)
        st.components.v1.html(html, height=600, scrolling = True)     

    # 3. Entity framing & timeline

    if not df_f.empty:
        df_f = df_f[df_f['main_role'].isin(role_filter)]

        st.header("3. Role Distribution & Transition Timeline")
        dist = df_f['main_role'].value_counts().reset_index()
        dist.columns = ['role','count']
        
        color_list = [ROLE_COLORS.get(role, "#cccccc") for role in dist['role']]
        domain_list = dist['role'].tolist()

        #chart        
        exploded = df_f.explode('fine_roles')
        grouped = exploded.groupby(['main_role', 'fine_roles']).size().reset_index(name='count')
        grouped = grouped.sort_values(by=['main_role', 'fine_roles'])

        # Compute the cumulative sum within each main_role
        grouped['cumsum'] = grouped.groupby('main_role')['count'].cumsum()
        grouped['prevsum'] = grouped['cumsum'] - grouped['count']
        grouped['entities'] = grouped['prevsum'] + grouped['count'] / 2

        # Bar chart
        bars = alt.Chart(grouped).mark_bar(stroke='black', strokeWidth=0.5).encode(
            x=alt.X('main_role:N', title='Main Role'),
            y=alt.Y('count:Q', stack='zero'),
            color=alt.Color('main_role:N', scale=alt.Scale(domain=domain_list, range=color_list), legend=None),
            tooltip=['main_role', 'fine_roles', 'count']
        )

        label_chart = alt.Chart(grouped).mark_text(
            color='black',
            fontSize=11
        ).encode(
            x='main_role:N',
            y=alt.Y('entities:Q'),  # <- exact center of the segment
            text='fine_roles:N'
        )

        # Combine
        chart = (bars + label_chart).properties(
            width=500,
            title='Main Roles with Fine-Grained Role Segments'
        )

        st.altair_chart(chart, use_container_width=True)

        #timeline
        timeline = alt.Chart(df_f).mark_bar().encode(
            x=alt.X('start:Q', title='Position'), x2='end:Q',
            y=alt.Y('entity:N', title='Entity'),
            color=alt.Color('main_role:N', scale=alt.Scale(domain=list(ROLE_COLORS.keys()), range=list(ROLE_COLORS.values()))),
            tooltip=['entity','main_role','confidence']
        ).properties(height=200)
        st.altair_chart(timeline, use_container_width=True)

        role_counts = df_f['main_role'].value_counts().reset_index()
        role_counts.columns = ['main_role', 'count']

        #pie chart
        pie = alt.Chart(role_counts).mark_arc(innerRadius=50).encode(
            theta=alt.Theta(field='count', type='quantitative'),
            color=alt.Color(field='main_role', type='nominal', scale=alt.Scale(domain=list(ROLE_COLORS.keys()), range=list(ROLE_COLORS.values()))),
            tooltip=['main_role', 'count']
        ).properties(title="Main Role Distribution")

        st.altair_chart(pie, use_container_width=True)

    # --- Sentence Display by Role with Adaptive Layout ---
    st.markdown("## 4. Sentences by Role Classification")

    df_f['main_role'] = df_f['main_role'].str.strip().str.title()
    df_f['fine_roles'] = df_f['fine_roles'].apply(lambda roles: [r.strip().title() for r in roles if r.strip()])
    df_f = df_f[df_f['main_role'].isin(ROLE_COLORS)]

    main_roles = sorted(df_f['main_role'].unique())
    multiple_roles = len(main_roles) > 1
    cols_per_row = 2 if multiple_roles else 1

    role_cols = [st.columns(cols_per_row) for _ in range((len(main_roles) + cols_per_row - 1) // cols_per_row)]

    for idx, role in enumerate(main_roles):
        col = role_cols[idx // cols_per_row][idx % cols_per_row]

        with col:
            role_df = df_f[df_f['main_role'] == role][['sentence', 'fine_roles']].copy()
            role_df['fine_roles'] = role_df['fine_roles'].apply(tuple)
            role_sentences = role_df.drop_duplicates()

            st.markdown(
                f"<div style='background-color:{ROLE_COLORS[role]}; "
                f"padding: 8px; border-radius: 6px; font-weight:bold;'>"
                f"{role} — {len(role_sentences)} labels"
                f"</div>",
                unsafe_allow_html=True
            )

            for sent in role_sentences['sentence'].unique():
                html_block = format_sentence_with_spans(sent, labels, threshold)
                st_html(html_block, height=80, scrolling=False)

            fine_df = df_f[df_f['main_role'] == role].explode('fine_roles')
            fine_df = fine_df[fine_df['fine_roles'].notnull() & (fine_df['fine_roles'] != '')]
            fine_roles = sorted(fine_df['fine_roles'].dropna().unique())

            if fine_roles and len(fine_roles)>1:
                selected_fine = st.selectbox(
                    f"Filter {role} by fine-grained role:",
                    ["Show all"] + fine_roles,
                    key=f"fine_{role}"
                )

                if selected_fine != "Show all":
                    fine_sents = fine_df[fine_df['fine_roles'] == selected_fine]['sentence'].drop_duplicates()
                    st.markdown(f"**{selected_fine}** — {len(fine_sents)} sentence(s):")
                    for s in fine_sents:
                        html_block = format_sentence_with_spans(s, labels, threshold)
                        st_html(html_block, height=80, scrolling=False)
            else: 
                for fine_role in fine_roles:
                    st.write(f"All annotations of this main role are of type: {fine_role}")




    # 4. Narrative classification
    st.header("#. Narrative Classification")
    df_n = predict_narrative_classification(article, threshold)
    st.dataframe(df_n)

    # 5. Free-form narrative extraction
    st.header("#. Free-form Narrative Extraction")
    st.write(extract_narrative(article))


    # 6. Bias identification & rewriting
    #st.header("6. Bias Identification & Rewriting")
    #mode_rw = st.radio("Rewrite mode", ["conservative","aggressive"])  
    #suggestions = identify_bias_and_rewrite(article, mode_rw)
    #for s in suggestions:
    #    st.write(f"**Span:** {s['span']}  \n**Suggestion:** {s['suggestion']}")




st.markdown("---")
st.markdown("*UGRIP 2025 FRaN-X Team* ")