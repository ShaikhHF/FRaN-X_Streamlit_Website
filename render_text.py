import streamlit as st
import pandas as pd
import re
from collections import defaultdict
import html as html_utils 
from collections import defaultdict
import html as html_utils

ROLE_COLORS = {
   "Protagonist": "#a1f4a1",
   "Antagonist":  "#f4a1a1",
   "Innocent":    "#a1c9f4",
}

def reformat_text_html_with_tooltips(text, labels_dict, hide_repeat=False, highlighted_word=None):
    spans = []
    entity_history = defaultdict(set)

    # Collect spans
    for entity, mentions in labels_dict.items():
        if not isinstance(mentions, list):
            st.stop()
            raise TypeError(f"Expected a list for entity '{entity}', but got {type(mentions)}")

        for mention in mentions:
            if not isinstance(mention, dict):
                st.stop()
                raise TypeError(f"Malformed mention at entity '{entity}': {mention}")

            start = mention.get("start_offset", 0)
            end = mention.get("end_offset", 0)

            start = max(0, min(start, len(text)))
            end = max(start, min(end, len(text)))

            main_role = mention.get("main_role", "")
            color = ROLE_COLORS.get(main_role, "#000000")
            fine_roles_set = frozenset(r.strip().title() for r in mention.get("fine_roles", []))
            fine_roles_str = ", ".join(fine_roles_set)

            is_repeated = fine_roles_set in entity_history[entity]
            adjusted_color = color
            if hide_repeat and is_repeated and color.startswith("#"):
                r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
                adjusted_color = f"rgba({r}, {g}, {b}, 0.3)"
            else:
                entity_history[entity].add(fine_roles_set)

            tooltip = (
                f"Role: {main_role or 'Unknown'}<br>"
                f"Confidence: {mention.get('confidence', 'N/A')}<br>"
                f"Fine roles: {fine_roles_str}"
            )

            entity_text = text[start:end].strip() or entity

            if highlighted_word:
                pattern = re.escape(highlighted_word)
                entity_text = re.sub(
                    rf"(?<![\w])({pattern})(?=\b|'s|s\b|\W)",
                    r'<span style="border: 2px solid black; background-color:yellow; padding:1px 4px; margin: 0 1px; border-radius: 4px;">\1</span>',
                    entity_text,
                    flags=re.IGNORECASE
                )

            spans.append({
                "start": start,
                "end": end,
                "html": (
                    f'<span class="entity" '
                    f'style="background-color:{adjusted_color}; padding:3px 6px; border-radius:4px;" '
                    f'data-tooltip="{tooltip}">'
                    f'{entity_text} | <span style="font-size: smaller;">{fine_roles_str}</span></span>'
                )
            })

    # Sort by start offset
    spans.sort(key=lambda x: x["start"])

    # Assemble annotated text with optional highlighting
    result = []
    last_idx = 0
    for span in spans:
        start, end = span["start"], span["end"]
        if start < last_idx:
            continue  # Overlap detected

        segment = text[last_idx:start]
        if highlighted_word:
            pattern = re.escape(highlighted_word)
            segment = re.sub(
                rf"(?<![\w])({pattern})(?=\b|'s|s\b|\W)",
                r'<span style="border: 2px solid black; background-color:yellow; padding:1px 4px; margin:0 1px; border-radius:4px;">\1</span>',
                segment,
                flags=re.IGNORECASE
            )
        result.append(segment)
        result.append(span["html"])
        last_idx = end

    # Handle the tail segment
    tail = text[last_idx:]
    if highlighted_word:
        pattern = re.escape(highlighted_word)
        tail = re.sub(
            rf"(?<![\w])({pattern})(?=\b|'s|s\b|\W)",
            r'<span style="border: 2px solid black; background-color:yellow; padding:1px 4px; margin:0 1px; border-radius:4px;">\1</span>',
            tail,
            flags=re.IGNORECASE
        )
    result.append(tail)

    annotated = ''.join(result)

    # Final HTML assembly
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




def predict_entity_framing(labels, threshold: float = 0.0):
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


def format_sentence_with_spans(sentence_text, labels, threshold, hide_repeat=True, show_fine_roles=False, seen_fine_roles=None):
    if seen_fine_roles is None:
        seen_fine_roles = defaultdict(set)

    spans = []
    sentence_lower = sentence_text.lower()

    for entity, mentions in labels.items():
        entity_key = entity.strip().lower()

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

            main_role = mention.get('main_role', '')
            base_color = ROLE_COLORS.get(main_role, "#000000")

            fine_roles_raw = mention.get('fine_roles', [])
            fine_roles_set = frozenset(r.strip().title() for r in fine_roles_raw)
            fine_roles_str = ", ".join(fine_roles_set)

            is_repeated = fine_roles_set in seen_fine_roles[entity_key]
            seen_fine_roles[entity_key].add(fine_roles_set)

            if hide_repeat and is_repeated and base_color.startswith("#") and len(base_color) == 7:
                r = int(base_color[1:3], 16)
                g = int(base_color[3:5], 16)
                b = int(base_color[5:7], 16)
                background_color = f"rgba({r}, {g}, {b}, 0.3)"
            else:
                background_color = base_color

            mention_display = html_utils.escape(sentence_text[match_start:match_end])

            if show_fine_roles:
                fine_roles_html = f' | <span style="font-size:smaller; opacity:0.75;">{fine_roles_str}</span>'
            else:
                fine_roles_html = ""

            span_html = (
                f'<span style="background-color:{background_color}; padding:3px 6px; border-radius:4px;">'
                f'{mention_display}{fine_roles_html}</span>'
            )

            spans.append({
                "start": match_start,
                "end": match_end,
                "html": span_html
            })

    # Merge spans
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
        '<style> body { font-family: sans-serif; } </style>'
        '</head><body>'
        '<div style="white-space: pre-wrap;">' + body + '</br> </div>'
        '</body></html>'
    )

    return full_html, seen_fine_roles