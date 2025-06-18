import streamlit as st
import pandas as pd
import altair as alt
import requests
from bs4 import BeautifulSoup
import re
from load_annotations import load_article, load_labels, load_file_names

TXT_FILE = 'chunk_data/EN_CC_100013.txt'
CSV_FILES = ['split_data/entry_0.csv', 'split_data/entry_1.csv']


ROLE_COLORS = {
    "Protagonist": "#a1f4a1",
    "Antagonist":  "#f4a1a1",
    "Innocent":    "#a1c9f4",
}

# Predict entity framing with span info
def predict_entity_framing(text, labels, threshold:float = 0.0):
    records = []
    for lbl in labels:
        pattern = re.escape(lbl['entity'])
        for m in re.finditer(rf"\b{pattern}\b", text):
            if lbl['confidence'] >= threshold:
                records.append({
                    'entity': lbl['entity'],
                    'main_role': lbl['main_role'],
                    'fine_roles': lbl['fine_roles'],
                    'confidence': lbl['confidence'],
                    'start': m.start(),
                    'end': m.end()
                })
        records.append({'entity': 'abcdefghijklmnopqrstuvwxyz','main_role': 'innocent','fine_roles': 'forgotten','confidence': '0.0','start': 0,'end': 0})#breaks when there is no entity
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

# Identify bias & suggestions
def identify_bias_and_rewrite(text, mode="conservative"):
    return [
        {"span": "British pressure",          "suggestion": "UK diplomatic influence"},
        {"span": "wage \u201cto the last Ukrainian\u201d", "suggestion": "intensify conflict"},
    ]

def annotate_article_html(text, df):
    annotated = text

    if not df.empty and 'entity' in df.columns:
        entities = df[df['entity'].apply(lambda x: isinstance(x, str) and x.strip() != "")]
        
        if not entities.empty:
            entities = entities.sort_values('entity', key=lambda s: s.str.len(), ascending=False)

            def escape_entity(entity):
                return re.sub(r'([.^$*+?{}\[\]\\|()])', r'\\\1', entity)

            entity_patterns = [
                rf'(?<!\w){escape_entity(ent)}(?!\w)'
                for ent in entities['entity']
            ]

            if entity_patterns:
                pattern = r'(' + '|'.join(entity_patterns) + r')'

                def replacer(m):
                    entity = m.group(0)
                    matches = entities[entities['entity'] == entity]
                    if matches.empty:
                        return entity  # fallback

                    row = matches.iloc[0]
                    color = ROLE_COLORS.get(row.get('main_role', ''), '#ffffb3')
                    fine_roles = row.get('fine_roles', [])
                    if isinstance(fine_roles, str):
                        try:
                            fine_roles = eval(fine_roles)
                        except:
                            fine_roles = [fine_roles]

                    tooltip = (
                        f"Role: {row.get('main_role', 'Unknown')}<br>"
                        f"Confidence: {row.get('confidence', 'N/A')}<br>"
                        f"Fine roles: {', '.join(fine_roles)}"
                    )
                    return (
                        f'<span class="entity" '
                        f'style="background-color:{color}; padding:3px 4px; border-radius:3px;" '
                        f'data-tooltip="{tooltip}">{entity}</span>'
                    )

                annotated = re.sub(pattern, replacer, annotated)

    # Always return full HTML structure even if there are no entities
    html = (
        '<html><head>'
        '<script src="https://unpkg.com/@popperjs/core@2"></script>'
        '<script src="https://unpkg.com/tippy.js@6"></script>'
        '<link rel="stylesheet" href="https://unpkg.com/tippy.js@6/animations/scale.css"/>'
        '<style> body { font-family: sans-serif; white-space: pre-wrap; } </style>'
        '</head><body>'
        + annotated.replace("\n", "<br>") +
        '<script>'
        'tippy(".entity", {'
        ' content: reference => reference.getAttribute("data-tooltip"),'  
        ' allowHTML: true,'
        ' trigger: "mouseenter click",'
        ' interactive: true,'
        ' animation: "scale"'
        '});'
        '</script>'
        '</body></html>'
    )
    return html


# --- Streamlit App ---
import streamlit.components.v1 as components

st.set_page_config(page_title="FRaN-X", layout="wide")
st.title("FRaN-X: Entity Framing & Narrative Analysis")


# Article input
st.header("1. Article Input")
use_example = st.checkbox("Use example article for illustration")

# Sidebar controls
st.sidebar.header("Settings")
if use_example:
    folder_path = 'chunk_data'
    file_names = load_file_names(folder_path)
else:
    folder_path = 'user_articles'
    file_names = load_file_names(folder_path)

article_name = st.sidebar.selectbox("Choose a file", file_names)
if not isinstance(article_name, str):
    article = ''
else:
    article = load_article(folder_path +'/'+ article_name)
    if use_example:
        labels = load_labels('split_data', article_name) 
    else:
        labels = load_labels('user_articles', article_name)


threshold    = st.sidebar.slider("Narrative confidence threshold", 0.0, 1.0, 0.5, 0.01)
role_filter  = st.sidebar.multiselect("Filter roles", list(ROLE_COLORS.keys()), default=list(ROLE_COLORS.keys()))
show_tl      = st.sidebar.checkbox("Show transition timeline", True)
show_annot   = st.sidebar.checkbox("Show annotated article view", True)



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
    # 2. Entity framing & timeline
    df_f = predict_entity_framing(article, labels, threshold)

    if not df_f.empty:
        df_f = df_f[df_f['main_role'].isin(role_filter)]

        st.header("2. Role Distribution & Transition Timeline")
        dist = df_f['main_role'].value_counts().reset_index()
        dist.columns = ['role','count']
        
        color_list = [ROLE_COLORS.get(role, "#cccccc") for role in dist['role']]
        domain_list = dist['role'].tolist()

        st.altair_chart(alt.Chart(dist).mark_bar().encode(x='role',y='count',color=alt.Color('role',scale=alt.Scale(domain=domain_list, range=color_list))), use_container_width=True)
        if show_tl:
            timeline = alt.Chart(df_f).mark_bar().encode(
                x=alt.X('start:Q', title='Position'), x2='end:Q',
                y=alt.Y('entity:N', title='Entity'),
                color=alt.Color('main_role:N', scale=alt.Scale(domain=list(ROLE_COLORS.keys()), range=list(ROLE_COLORS.values()))),
                tooltip=['entity','main_role','confidence']
            ).properties(height=200)
            st.altair_chart(timeline, use_container_width=True)

    # 3. Annotated article view
    if show_annot:
        st.header("3. Framing-Annotated Article View")
        html = annotate_article_html(article, df_f)
        components.html(html, height=600)

    # 4. Narrative classification
    st.header("4. Narrative Classification")
    df_n = predict_narrative_classification(article, threshold)
    st.dataframe(df_n)

    # 5. Free-form narrative extraction
    st.header("5. Free-form Narrative Extraction")
    st.write(extract_narrative(article))

    # 6. Bias identification & rewriting
    st.header("6. Bias Identification & Rewriting")
    mode_rw = st.radio("Rewrite mode", ["conservative","aggressive"])  
    suggestions = identify_bias_and_rewrite(article, mode_rw)
    for s in suggestions:
        st.write(f"**Span:** {s['span']}  \n**Suggestion:** {s['suggestion']}")




st.markdown("---")
st.markdown("*UGRIP 2025 FRaN-X Team* ")