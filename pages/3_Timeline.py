import streamlit as st
import pandas as pd
from sidebar import render_sidebar, ROLE_COLORS
from render_text import predict_entity_framing

st.set_page_config(page_title="FRaN-X", layout="wide")
st.title("In-Depth Timeline")
st.write("See how each entity changes its main role and fine grain role over time")

article, labels, user_folder, threshold, role_filter, hide_repeat = render_sidebar()


def highlight_fine_roles(sentence, roles, color):
   if not isinstance(roles, list):
       roles = [roles]
   for role in roles:
       if role and isinstance(role, str) and role in sentence:
           sentence = sentence.replace(
               role,
               f"<span style='background-color:{color}; padding:2px 6px; border-radius:4px; font-weight:bold;'>{role}</span>"
           )
   return sentence


def render_block(block, role_main, role_fine, count, color):
   fine_display = ", ".join(role_fine) if isinstance(role_fine, list) else role_fine

   with st.expander(f"{role_main} / {fine_display} ({count} instances)", expanded=True):
       for b in block:
           fine_roles = b['fine_roles'] if isinstance(b['fine_roles'], list) else [b['fine_roles']]
           highlighted_sentence = highlight_fine_roles(b['sentence'], fine_roles, color)
           highlighted_roles = ", ".join(
               f"<span style='background-color:{color}; padding:2px 6px; border-radius:4px; font-weight:bold;'>{fr}</span>"
               for fr in fine_roles if isinstance(fr, str)
           )
           st.markdown(
               f"""<div style='padding:10px; border-radius:5px; margin-bottom:5px; border: 1px solid #ddd;'>
               <b>Fine Role(s):</b> {highlighted_roles}<br>
               <b>Confidence:</b> {b['confidence']:.2f}<br>
               <b>Sentence:</b> {highlighted_sentence}</div>""",
               unsafe_allow_html=True
           )


if article and labels:
   df_f = predict_entity_framing(labels, threshold)
   df_f = df_f[df_f['main_role'].isin(role_filter)]

   df_f.sort_values(by=["entity", "start"], inplace=True)
   entity_order = df_f.groupby("entity")["start"].min().sort_values().index.tolist()

   for entity in entity_order:
    group = df_f[df_f["entity"] == entity]
    total_annotations = len(group)
    st.divider()
    st.markdown(f"#### {entity} (total: {total_annotations})")

    block = []
    prev_main = prev_fine = None
    count = 0

    for _, row in group.iterrows():
        main = row["main_role"]
        fine = row["fine_roles"] if isinstance(row["fine_roles"], list) else [row["fine_roles"]]

        if (main, fine) != (prev_main, prev_fine) and prev_main is not None:
            # Insert role change indicator

            color = ROLE_COLORS.get(prev_main, "#ccc")
            render_block(block, prev_main, prev_fine, count, color)

            prev_fine_display = ", ".join(prev_fine) if isinstance(prev_fine, list) else prev_fine
            new_fine_display = ", ".join(fine) if isinstance(fine, list) else fine
            st.markdown(f"""
            <div style='background-color:rgba(249, 249, 249, 0.7); border-left:4px solid {ROLE_COLORS.get(main)}; padding:10px; margin:10px 0; border-radius:6px; font-size:15px;'>
            <b>↪️ Role Change {entity}</b><br>
            <b>From:</b> {prev_main} / {prev_fine_display}<br>
            <b>To:</b> {main} / {new_fine_display}
            </div>
            """, unsafe_allow_html=True)


            block = []
            count = 0

        block.append(row)
        count += 1
        prev_main, prev_fine = main, fine

    if block:
        color = ROLE_COLORS.get(prev_main, "#ccc")
        render_block(block, prev_main, prev_fine, count, color)

st.markdown("---")
st.markdown("*UGRIP 2025 FRaN-X Team* ")