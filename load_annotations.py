import csv
from collections import defaultdict
import os
import random #remove later once the confidence column is added
from pathlib import Path
import streamlit as st


def capitalize(s):
    return s.capitalize() if s else s

def load_article(filepath):
    if filepath.endswith("Select a file"):
        return ""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read().strip()

"""
def load_labels(folder_name, article_file_name, threshold):
    label_map = defaultdict(lambda: {
        'entity': '',
        'main_role': '',
        'fine_roles': set(),
        'confidence': 0.0
    })

    csv_files = [str(f) for f in Path(folder_name).iterdir() if f.is_file()]

    for csv_file in csv_files:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                c = round(random.uniform(0.6, 0.95), 2)
                if c > threshold:
                    
                    if article_file_name == row['article_id']:
                        entity = row['entity_mention']
                        role_class = capitalize(row['main_role'])
                        fine_role = row['fine_grained_roles'][2:-2].replace("'", "")
                        #confidence = row['confidence']
                        confidence = c

                        key = entity
                        if label_map[key]['entity'] == '':
                            label_map[key]['entity'] = entity
                            label_map[key]['main_role'] = role_class
                        label_map[key]['fine_roles'].add(fine_role)
                        label_map[key]['confidence'] = max(label_map[key]['confidence'], confidence)

    # Convert sets to sorted lists
    for label in label_map.values():
        label['fine_roles'] = sorted(label['fine_roles'])

    return list(label_map.values())
"""

import csv
import random
from collections import defaultdict
from pathlib import Path

def capitalize(s):
    return s[0].upper() + s[1:] if s else s

def load_labels(folder_name, article_file_name, threshold):
    role_timeline = defaultdict(list)
    csv_files = [str(f) for f in Path(folder_name).iterdir() if f.is_file()]

    for csv_file in csv_files:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['article_id'] != article_file_name:
                    continue

                confidence = round(random.uniform(0.6, 0.95), 2)
                if confidence < threshold:
                    continue

                entity = row['entity_mention'].strip()
                main_role = capitalize(row['main_role'].strip())

                raw_roles = row['fine_grained_roles'].strip()
                fine_roles = [r.strip() for r in raw_roles[2:-2].replace("'", "").split(",") if r.strip()]

                try:
                    start_offset = int(row['start_offset'])
                    end_offset = int(row['end_offset'])
                except ValueError:
                    continue  # skip bad data

                role_entry = {
                    'start_offset': start_offset,
                    'end_offset': end_offset,
                    'main_role': main_role,
                    'fine_roles': sorted(fine_roles),
                    'confidence': confidence
                }

                role_timeline[entity].append(role_entry)

    for entity in role_timeline:
        if isinstance(role_timeline[entity], dict):
            role_timeline[entity] = [role_timeline[entity]]

    # Sort mentions by start_offset
    for mentions in role_timeline.values():
        mentions.sort(key=lambda x: x['start_offset'])

    return dict(role_timeline)


def load_file_names(folder_path):
    files = os.listdir(folder_path)
    return tuple(files)
