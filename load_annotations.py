import csv
from collections import defaultdict
import os
from pathlib import Path

def capitalize(s):
    return s.capitalize() if s else s

def load_article(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read().strip()

def load_labels(folder_name, article_file_name):
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
                if article_file_name == row['article_id']:
                    entity = row['entity_mention']
                    role_class = capitalize(row['main_role'])
                    fine_role = row['fine_grained_roles'][2:-2].replace("'", "")
                    confidence = 0.7

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


def load_file_names(folder_path):
    files = os.listdir(folder_path)
    return tuple(files)
