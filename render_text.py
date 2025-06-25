import streamlit as st
import pandas as pd
from streamlit.components.v1 import html as st_html
import re

ROLE_COLORS = {
   "Protagonist": "#a1f4a1",
   "Antagonist":  "#f4a1a1",
   "Innocent":    "#a1c9f4",
}

def reformat_text_html_with_tooltips(text, labels_dict, highlighted_word = None):
    spans = []
    for entity, mentions in labels_dict.items():
        if not isinstance(mentions, list):
            st.stop()
            raise TypeError(f"Expected a list for entity '{entity}', but got {type(mentions)}")
      
        for i, mention in enumerate(mentions):
            if not isinstance(mention, dict):
                st.stop()
                raise TypeError(f"Malformed mention at entity '{entity}', index {i}: {mention}")
          
            start = mention.get("start_offset", 0)
            end = mention.get("end_offset", 0)
      
            # Clip to valid range
            start = max(0, min(start, len(text)))
            end = max(start, min(end, len(text)))
  
            # Ensure end ≥ start
            color = ROLE_COLORS.get(mention.get("main_role", ""), "#000000")
            fine_roles = ", ".join([r.strip().title() for r in mention.get("fine_roles", [])])
            tooltip = (
                f"Role: {mention.get('main_role', 'Unknown')}<br>"
                f"Confidence: {mention.get('confidence', 'N/A')}<br>"
                f"Fine roles: {fine_roles}"
            )
            entity_text = text[start:end].strip() or entity # fallback if empty
          
            spans.append({
                "start": start,
                "end": end,
                "html": (
                    f'<span class="entity" '
                    f'style="background-color:{color}; padding:3px 6px; border-radius:4px;" '
                    f'data-tooltip="{tooltip}">'
                    f'{entity_text} | <span style="font-size: smaller;">{fine_roles}</span></span>'
                    )
                })
              
    # Sort spans by start index
    spans.sort(key=lambda x: x["start"])
      
    result = []
    last_idx = 0
    for span in spans:
        start, end = span["start"], span["end"]
        if start < last_idx:
            # Overlap detected, log it (optional)
            continue # skip or resolve overlap if needed
        result.append(text[last_idx:start])
        result.append(span["html"])
        last_idx = end
              
    result.append(text[last_idx:]) # ALWAYS append tail
      
    annotated = ''.join(result)
      
    html = (
        '<html><head>'
        '<script src="https://unpkg.com/@popperjs/core@2"></script>'
        '<script src="https://unpkg.com/tippy.js@6"></script>'
        '<link rel="stylesheet" href="https://unpkg.com/tippy.js@6/animations/scale.css"/>'
        '<style> body { font-family: sans-serif; } </style>'
        '</head><body>'
        '<div style="white-space: pre-wrap;">'
        + annotated +
        '</div>'
        '<script>'
        'tippy(".entity", {'
        'content: reference => reference.getAttribute("data-tooltip"),'
        'allowHTML: true,'
        ' trigger: "mouseenter click",'
        ' interactive: true,'
        ' animation: "scale"'
        '});'
        '</script>'
        '</body></html>'
    )

    if highlighted_word:
        pattern = re.escape(highlighted_word)
        html = re.sub(
            rf"(?<![\w])({pattern})(?=\b|'s|s\b|\W)",
            r'<span style="border: 2px solid black; background-color:yellow; padding:1px 4px; margin: 0 1px; border-radius: 4px;">\1</span>',
            html,
            flags=re.IGNORECASE
        )
  
    return html

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
                    'end': mention['end_offset'],
                    'sentence':mention['sentence']
                })

    # Ensure there is at least one record to avoid breaking downstream code
    if not records:
        records.append({
            'entity': 'abcdef',
            'main_role': 'innocent',
            'fine_roles': ['forgotten'],
            'confidence': 0.0,
            'start': 0,
            'end': 0,
            'sentence':'abcdef'
        })

    return pd.DataFrame(records)



import html as html_utils 

def format_sentence_with_spans(sentence_text, labels, threshold, show_fine_roles=False):
    spans = []
    used_spans = []

    sentence_lower = sentence_text.lower()

    for entity, mentions in labels.items():
        for mention in mentions:
            if mention.get('confidence', 0) < threshold:
                continue
            if mention.get('sentence', '').strip() != sentence_text.strip():
                continue

            mention_text = entity.strip()
            if not mention_text:
                continue

            match_start = sentence_lower.find(mention_text.lower())
            if match_start == -1:
                continue

            match_end = match_start + len(mention_text)

            if any(match_start < end and match_end > start for start, end in used_spans):
                continue

            used_spans.append((match_start, match_end))

            color = ROLE_COLORS.get(mention.get('main_role', ''), "#000000")
            fine_roles = ", ".join([r.strip().title() for r in mention.get('fine_roles', [])])

            if not show_fine_roles:
                span_html = (
                    f'<span style="background-color:{color}; padding:3px 6px; border-radius:4px;">'
                    f'{html_utils.escape(sentence_text[match_start:match_end])}'
                    f'<span style="font-size:smaller; opacity:0.75;"> </span>'
                    f'</span>'
                )
            else:
                span_html = (
                    f'<span style="background-color:{color}; padding:3px 6px; border-radius:4px;">'
                    f'{html_utils.escape(sentence_text[match_start:match_end])}'
                    f' |<span style="font-size:smaller; opacity:0.75;"> {fine_roles} </span>'
                    f'</span>'
                )

            spans.append({
                "start": match_start,
                "end": match_end,
                "html": span_html
            })

    spans.sort(key=lambda x: x["start"])
    result = []
    last_idx = 0
    for span in spans:
        start = span["start"]
        end = span["end"]
        if start < last_idx:
            continue
        result.append(html_utils.escape(sentence_text[last_idx:start]))
        result.append(span["html"])
        last_idx = end
    result.append(html_utils.escape(sentence_text[last_idx:]))

    body = ''.join(result)

    full_html = (
        '<html><head>'
        '<link rel="stylesheet" href="https://unpkg.com/tippy.js@6/animations/scale.css"/>'
        '<style> body { font-family: sans-serif; } </style>'
        '</head><body>'
        '<div style="white-space: pre-wrap; overflow: visible; position: relative; z-index: 0;">' +
        body +
        '</div>'
    )

    return full_html
