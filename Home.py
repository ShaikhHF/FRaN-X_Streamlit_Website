import streamlit as st
import pandas as pd
import altair as alt
import requests
from bs4 import BeautifulSoup
import re
from sidebar import render_sidebar


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
        records.append({'entity': 'abcdef','main_role': 'innocent','fine_roles': 'forgotten','confidence': '0.0','start': 0,'end': 0})#breaks when there is no entity
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

def escape_entity(entity):
    return re.sub(r'([.^$*+?{}\[\]\\|()])', r'\\\1', entity)

def reformat_text_html_with_tooltips(text, labels):
    annotated = text

    if labels:
        labels_sorted = sorted(labels, key=lambda d: len(d['entity']), reverse=True)

        entity_patterns = [
            rf'(?<!\w){escape_entity(lbl["entity"])}(?!\w)'
            for lbl in labels_sorted
            if isinstance(lbl.get("entity"), str) and lbl["entity"].strip()
        ]

        if entity_patterns:
            pattern = r'(' + '|'.join(entity_patterns) + r')'

            def replacer(m):
                entity = m.group(0)
                match = next((lbl for lbl in labels_sorted if lbl["entity"] == entity), None)
                if not match:
                    return entity

                color = ROLE_COLORS.get(match.get("main_role", ""), "#000000")

                # Clean fine_roles
                fine_roles_raw = match.get("fine_roles", [])
                if isinstance(fine_roles_raw, str):
                    try:
                        fine_roles_raw = eval(fine_roles_raw)
                    except:
                        fine_roles_raw = [fine_roles_raw]

                cleaned_roles = [
                    r.strip(' "\'').title()
                    for r in fine_roles_raw
                    if isinstance(r, str)
                ]
                fine_roles = ", ".join(cleaned_roles)

                tooltip = (
                    f"Role: {match.get('main_role', 'Unknown')}<br>"
                    f"Confidence: {match.get('confidence', 'N/A')}<br>"
                    f"Fine roles: {fine_roles}"
                )

                return (
                    f'<span class="entity" '
                    f'style="background-color:{color}; padding:3px 6px; border-radius:4px;" '
                    f'data-tooltip="{tooltip}">'
                    f'{entity} | <span style="font-size: smaller;">{fine_roles}</span></span>'
                )


            annotated = re.sub(pattern, replacer, annotated)

    # Wrap in full HTML document with Tippy support
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


article, labels, use_example, threshold, role_filter = render_sidebar()





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
    show_tl = st.checkbox("Show transition timeline", True)

    if not df_f.empty:
        df_f = df_f[df_f['main_role'].isin(role_filter)]

        st.header("3. Role Distribution & Transition Timeline")
        dist = df_f['main_role'].value_counts().reset_index()
        dist.columns = ['role','count']
        
        color_list = [ROLE_COLORS.get(role, "#cccccc") for role in dist['role']]
        domain_list = dist['role'].tolist()

        #chart
        #st.altair_chart(alt.Chart(dist).mark_bar().encode(x='role',y='count',color=alt.Color('role',scale=alt.Scale(domain=domain_list, range=color_list))), use_container_width=True)
        
        exploded = df_f.explode('fine_roles')
        grouped = exploded.groupby(['main_role', 'fine_roles']).size().reset_index(name='count')
        grouped = grouped.sort_values(by=['main_role', 'fine_roles'])

        # Compute the cumulative sum within each main_role
        grouped['cumsum'] = grouped.groupby('main_role')['count'].cumsum()
        grouped['prevsum'] = grouped['cumsum'] - grouped['count']
        grouped['midpoint'] = grouped['prevsum'] + grouped['count'] / 2

        # Color mapping
        domain_list = list(ROLE_COLORS.keys())
        color_list = [ROLE_COLORS.get(role, "#cccccc") for role in domain_list]

        # Bar chart
        bars = alt.Chart(grouped).mark_bar(stroke='black', strokeWidth=0.5).encode(
            x=alt.X('main_role:N', title='Main Role'),
            y=alt.Y('count:Q', stack='zero'),
            color=alt.Color('main_role:N', scale=alt.Scale(domain=domain_list, range=color_list), legend=None),
            tooltip=['main_role', 'fine_roles', 'count']
        )

        # Label chart — use precomputed 'midpoint' for y positioning
        labels = alt.Chart(grouped).mark_text(
            color='black',
            fontSize=11
        ).encode(
            x='main_role:N',
            y=alt.Y('midpoint:Q'),  # <- exact center of the segment
            text='fine_roles:N'
        )

        # Combine
        chart = (bars + labels).properties(
            width=500,
            title='Main Roles with Fine-Grained Role Segments'
        )

        st.altair_chart(chart, use_container_width=True)

        #timeline
        if show_tl:
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
    st.header("6. Bias Identification & Rewriting")
    mode_rw = st.radio("Rewrite mode", ["conservative","aggressive"])  
    suggestions = identify_bias_and_rewrite(article, mode_rw)
    for s in suggestions:
        st.write(f"**Span:** {s['span']}  \n**Suggestion:** {s['suggestion']}")


    



st.markdown("---")
st.markdown("*UGRIP 2025 FRaN-X Team* ")