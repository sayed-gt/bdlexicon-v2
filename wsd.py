import os
import ast
import bkit
import time
import random
import argparse
import warnings
import polars as pl
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from joblib import Parallel, delayed
from functools import cache, lru_cache
from bkit.transform import normalize_punctuation_spaces, clean_text
from bkit.lemmatizer import lemmatize_word, lemmatize

# Timer decorator
def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper


# Define the function to find sentences for a word
def balanced_matching(word: str, df: pd.DataFrame) -> dict:
    word = word.strip()
    lemma = ast.literal_eval(lemmatize(f"['{word}']"))[0].strip()

    word_df = df[df["original"].apply(lambda sentence: word in sentence.split())]
    lemma_df = df[df["lemmatized"].str.contains(fr"\b{lemma}\b", regex=True)]
    
    # Combine both matches, ensuring no duplicate rows
    combined_df = pd.concat([word_df, lemma_df]).drop_duplicates().head(1)
    final_dict = {
        "sentence": combined_df["original"].tolist(),
        "url": combined_df["url"].tolist(),
        "doc_id": combined_df["doc_id"].tolist(),
        "text_id": combined_df["text_id"].tolist(),
        "categories": combined_df["categories"].tolist(),
        "domain_type": combined_df["domain_type"].tolist(),
        "domain_name": combined_df["domain_name"].tolist()
    }
    return final_dict


def fetch_from_multiple_jsons(json_files_path: str, word: str, max_sentences: int) -> dict:
    json_file_list = os.listdir(json_files_path)
    json_file_list = [json_file for json_file in json_file_list if json_file.endswith(".parquet")]
    random.shuffle(json_file_list)  # Shuffle the list to ensure random selection

    dictionary = {
        "sentence": [],
        "url": [],
        "doc_id": [],
        "text_id": [],
        "categories": [],
        "domain_type": [],
        "domain_name": [],
    }

    for json_file in json_file_list:
        try:
            sentence_df = pl.read_parquet(os.path.join(json_files_path, json_file)).to_pandas()
            if "original" not in sentence_df.columns:
                continue
        except Exception as e:
            print(f"Skipping {json_file}: {e}")
            continue

        return_dict = balanced_matching(word, sentence_df)

        # Add data while ensuring the total count doesn't exceed `max_sentences`
        for i in range(len(return_dict["sentence"])):
            if len(dictionary["sentence"]) < max_sentences:
                dictionary["sentence"].append(return_dict["sentence"][i])
                dictionary["url"].append(return_dict["url"][i])
                dictionary["doc_id"].append(return_dict["doc_id"][i])
                dictionary["text_id"].append(return_dict["text_id"][i])
                dictionary["categories"].append(return_dict["categories"][i])
                dictionary["domain_type"].append(return_dict["domain_type"][i])
                dictionary["domain_name"].append(return_dict["domain_name"][i])
            else:
                break

        # Break the loop if enough data is collected
        if len(dictionary["sentence"]) >= max_sentences:
            break

    dictionary["sentence"] = [normalize_punctuation_spaces(sentence) for sentence in dictionary["sentence"]]
    dictionary["sentence"] = [clean_text(sentence) + " ।" for sentence in dictionary["sentence"]]
    return dictionary


def group_senses(input_df: pd.DataFrame) -> pd.DataFrame:
    # Group by 'word' and aggregate 'sense' into lists
    grouped_df = input_df.groupby('word', as_index=False).agg({'sense': list})
    return grouped_df


@timer
def parallel_processed_dataframe(word_csv_path: str, json_dir: str, max_sentences: int, n_jobs: int) -> pd.DataFrame:
    # Read CSV into a pandas DataFrame
    df = pl.read_csv(word_csv_path).to_pandas()
    df = group_senses(df)

    # Parallel processing
    results = Parallel(n_jobs=n_jobs)(
        delayed(fetch_from_multiple_jsons)(
            json_files_path=json_dir,
            word=row["word"],
            max_sentences=max_sentences,
        ) for _, row in tqdm(df.iterrows(), total=len(df))
    )

    df["matched"] = results
    df["examples"] = df["matched"].apply(lambda x: x["sentence"])
    df["example_urls"] = df["matched"].apply(lambda x: x["url"])
    df["doc_id"] = df["matched"].apply(lambda x: x["doc_id"])
    df["text_id"] = df["matched"].apply(lambda x: x["text_id"])
    df["categories"] = df["matched"].apply(lambda x: x["categories"])
    df["domain_type"] = df["matched"].apply(lambda x: x["domain_type"])
    df["domain_name"] = df["matched"].apply(lambda x: x["domain_name"])
    df = df.drop(columns=["matched"])
    return df



parser = argparse.ArgumentParser()
parser.add_argument("--word_csv_path", type=str, required=True, help="Path to the CSV file containing the words and their corresponding lemmas.")
parser.add_argument("--json_files_path", type=str, required=True, help="Path to the directory containing the JSON files.")
parser.add_argument("--max_sentences", type=int, default=20)
parser.add_argument("--output_path", type=str, default=None)
parser.add_argument("--n_jobs", type=int, default=-1, help="Number of cores to use for parallel processing.")
args = parser.parse_args()


@timer
def main():
    word_csv_path = args.word_csv_path
    json_files_path = args.json_files_path
    max_sentences = args.max_sentences
    output_path = args.output_path
    n_jobs = args.n_jobs

    if output_path is None:
        output_path = f"{word_csv_path.removesuffix('.csv')}_repaired.csv"

    df = parallel_processed_dataframe(word_csv_path, json_files_path, max_sentences, n_jobs)
    df.to_csv(output_path, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()