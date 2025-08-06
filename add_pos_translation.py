import requests
import pandas as pd
import os
from tqdm import tqdm

tqdm.pandas()
from deep_translator import GoogleTranslator
translator = GoogleTranslator(source='bn', target='en')


pos_api = "http://182.163.99.83:9300/v1/POS/infer_pos"

def get_pos(text):
    payload = {"text": text}
    response = requests.post(pos_api, json=payload)
    json_data = response.json()
    return [d["label"] for d in json_data["predictions"]["entities"]]

def add_pos(df):
    df["pos"] = df["word"].progress_apply(get_pos)
    return df

def get_translation(word):
    try:
        result = translator.translate(word)  # from Bengali to English
        return result
    except Exception as e:
        print(f"Translation error for '{word}': {e}")
        return ""

def add_translation(df):
    df["translation"] = df["word"].progress_apply(get_translation)
    return df

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    data_dir = os.listdir(args.input_dir)
    for file in data_dir:
        df = pd.read_csv(os.path.join(args.input_dir, file))
        df = add_pos(df)
        #df = add_translation(df)
        df.to_csv(os.path.join(args.output_dir, file), index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)

