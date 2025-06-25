import streamlit as st
import time
import requests
from bs4 import BeautifulSoup
from sidebar import render_sidebar
import os

st.set_page_config(page_title="FRaN-X", initial_sidebar_state='expanded', layout="wide")
st.title("Upload Articles")
st.write("Choose a User Folder in the Sidebar to continue")

article, labels, user_folder, threshold, role_filter, hide_repeat = render_sidebar(True, False, True, True)

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
            filepath = os.path.join('user_articles', user_folder, safe_filename)

            # Check for duplicate filenames
            if os.path.exists(filepath):
                st.error(f"A file named '{safe_filename}' already exists. Please choose a different name.")
            else:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(article)
                st.success(f"Article saved as {safe_filename}. Please wait while your entity framing is being calculated")
                    
                time.sleep(2) # adjust once backend connected
                    
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



st.markdown("---")
st.markdown("*UGRIP 2025 FRaN-X Team* ")