import pandas as pd
import os


def read_stage1_output(filepath: str) -> pd.DataFrame:
    """
    Reads Stage 1 output in txt format and returns a DataFrame.
    Expected format per line:
    article_id<TAB>entity<TAB>start_offset<TAB>end_offset<TAB>main_role
    """
    records = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            article_id, entity, start, end, main_role = parts[:5]
            records.append({
                "article_id": article_id,
                "entity_mention": entity,
                "start_offset": int(start),
                "end_offset": int(end),
                "p_main_role": main_role,
            })
    return pd.DataFrame(records)


def attach_article_text(df: pd.DataFrame, articles_folder: str) -> pd.DataFrame:
    """
    Maps article_id to full article text. Assumes article files are in articles_folder.
    """
    articles = {}
    for file in os.listdir(articles_folder):
        if file.endswith(".txt"):
            with open(os.path.join(articles_folder, file), encoding="utf-8") as f:
                articles[file] = f.read()

    df["text"] = df["article_id"].map(articles)
    return df


if __name__ == "__main__":
    stage1_file = "predictions/filler-predicted.txt"
    article_folder = "uploaded_articles"  # where "filler.txt" is stored

    df = read_stage1_output(stage1_file, lang)
    df = attach_article_text(df, article_folder)

    df.to_csv("stage1_converted.csv", index=False, encoding="utf-8")
    print("✅ stage1_converted.csv saved.")
