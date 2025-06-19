import streamlit as st
import pandas as pd
import altair as alt
import requests
from bs4 import BeautifulSoup
import re
from sidebar import render_sidebar
from render_text import reformat_text_html_with_tooltips


ROLE_COLORS = {
   "Protagonist": "#a1f4a1",
   "Antagonist":  "#f4a1a1",
   "Innocent":    "#a1c9f4",
}

def predict_entity_framing(text, labels, threshold: float = 0.0):
    records = []

    for entity, mentions in labels.items():
        for mention in mentions:
            if mention['confidence'] >= threshold:
                records.append({
                    'entity': entity,
                    'main_role': mention['main_role'],
                    'fine_roles': mention['fine_roles'],
                    'confidence': mention['confidence'],
                    'start': mention['start_offset'],
                    'end': mention['end_offset']
                })

    # Ensure there is at least one record to avoid breaking downstream code
    if not records:
        records.append({
            'entity': 'abcdef',
            'main_role': 'innocent',
            'fine_roles': ['forgotten'],
            'confidence': 0.0,
            'start': 0,
            'end': 0
        })

    return pd.DataFrame(records)


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

#st.write(article)

if use_example:
    st.text_area("Example Article", article, height=300)
else:
    mode = st.radio("Input mode", ["Paste Text","URL"])
    if mode == "Paste Text":
        article = st.text_area("Article text", height=200)
    else:
        url = st.text_input("Article URL")
        article = ""
        if url:
            resp = requests.get(url)
            soup = BeautifulSoup(resp.content, 'html.parser')
            article = '\n'.join(p.get_text() for p in soup.find_all('p'))

if article:
    show_annot   = st.checkbox("Show annotated article view", True)
    df_f = predict_entity_framing(article, labels, threshold)

    # 3. Annotated article view
    if show_annot:
        html = reformat_text_html_with_tooltips(article, labels)
        st.components.v1.html(html, height=600)     

    # 2. Entity framing & timeline

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

        labels = alt.Chart(grouped).mark_text(
            color='black',
            fontSize=11
        ).encode(
            x='main_role:N',
            y=alt.Y('entities:Q'),  # <- exact center of the segment
            text='fine_roles:N'
        )

        # Combine
        chart = (bars + labels).properties(
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

    # 4. Narrative classification
    st.header("4. Narrative Classification")
    df_n = predict_narrative_classification(article, threshold)
    st.dataframe(df_n)

    # 5. Free-form narrative extraction
    st.header("5. Free-form Narrative Extraction")
    st.write(extract_narrative(article))


    # 6. Bias identification & rewriting
    #st.header("6. Bias Identification & Rewriting")
    #mode_rw = st.radio("Rewrite mode", ["conservative","aggressive"])  
    #suggestions = identify_bias_and_rewrite(article, mode_rw)
    #for s in suggestions:
    #    st.write(f"**Span:** {s['span']}  \n**Suggestion:** {s['suggestion']}")

    



st.markdown("---")
st.markdown("*UGRIP 2025 FRaN-X Team* ")