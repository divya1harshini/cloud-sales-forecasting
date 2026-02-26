import pandas as pd

def load_data(file):
    if file.filename.endswith('.csv'):
        return pd.read_csv(file)
    elif file.filename.endswith('.xlsx'):
        return pd.read_excel(file)
    elif file.filename.endswith('.json'):
        return pd.read_json(file)
    elif file.filename.endswith('.txt'):
        return pd.read_csv(file, delimiter='\t')
    else:
        return None
def clean_data(df):
    df = df.drop_duplicates()
    df = df.fillna(df.mean(numeric_only=True))
    return df
import re

def clean_text(text):
    text = re.sub(r'\W+', ' ', text)
    text = text.lower()
    return text