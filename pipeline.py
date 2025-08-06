import os
import ast
import bkit
import json
import hashlib
import uuid
import time
import random
import warnings
import pandas as pd
import polars as pl
from tqdm import tqdm
from joblib import Parallel, delayed
from functools import cache, lru_cache
from bkit.transform import normalize_punctuation_spaces, clean_text


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
def balanced_matching(word: str, lemma: str, df: pd.DataFrame) -> dict:
    word = word.strip()
    lemma = lemma.strip()

    word_df = df[df["original"].apply(lambda sentence: word in sentence.split())]
    lemma_df = df[df["lemmatized"].apply(lambda sentence: lemma in sentence.split())]
    
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


def unified_matching(word: str, lemma: str, sentence_df: pd.DataFrame) -> dict:
    word = word.strip()
    lemma = lemma.strip()
    matched_df = sentence_df[
        sentence_df["original"].apply(lambda sentence: word in sentence.split()) |
        sentence_df["lemmatized"].apply(lambda sentence: lemma in sentence.split()) |
        sentence_df["original"].str.contains(fr"\b{lemma}\b", regex=True)
    ]

    matched_df = matched_df.head(1)
    final_dict = {
        "sentence": matched_df["original"].tolist(),
        "url": matched_df["url"].tolist(),
        "doc_id": matched_df["doc_id"].tolist(),
        "text_id": matched_df["text_id"].tolist(),
        "categories": matched_df["categories"].tolist(),
        "domain_type": matched_df["domain_type"].tolist(),
        "domain_name": matched_df["domain_name"].tolist()
    }
    return final_dict


def fetch_from_multiple_jsons(json_files_path: str, word: str, lemma: str, max_sentences: int, matching_function: str) -> dict:
    matching_function = unified_matching if matching_function == "unified" else balanced_matching
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

        return_dict = matching_function(word, lemma, sentence_df)

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



@timer
def processed_dataframe(word_csv_path: str, json_dir: str, max_sentences: int, matching_function: str) -> pd.DataFrame:
    df = pl.read_csv(word_csv_path).to_pandas()
    
    # Initialize an empty list to store results
    matched_sentences = []
    # Iterate through rows with a progress bar
    for index, row in tqdm(df.iterrows(), total=len(df)):
        result = fetch_from_multiple_jsons(
            word=row["word"], 
            lemma=row["lemma"],
            json_files_path=json_dir, 
            max_sentences=max_sentences,
            matching_function=matching_function
        )
        matched_sentences.append(result)

    # Add results back to the DataFrame
    df["matched"] = matched_sentences
    df["examples"] = df["matched"].apply(lambda x: x["sentence"])
    df["example_urls"] = df["matched"].apply(lambda x: x["url"])
    df["doc_id"] = df["matched"].apply(lambda x: x["doc_id"])
    df["text_id"] = df["matched"].apply(lambda x: x["text_id"])
    df["categories"] = df["matched"].apply(lambda x: x["categories"])
    df["domain_type"] = df["matched"].apply(lambda x: x["domain_type"])
    df["domain_name"] = df["matched"].apply(lambda x: x["domain_name"])
    df = df.drop(columns=["matched"])
    return df

@timer
def parallel_processed_dataframe(word_csv_path: str, json_dir: str, max_sentences: int, matching_function: str, n_jobs: int) -> pd.DataFrame:
    df = pd.read_csv(word_csv_path)

    # Ensure required columns exist before using apply()
    for col in ["examples", "example_urls", "doc_id", "text_id", "categories", "domain_type", "domain_name"]:
        if col not in df.columns:
            df[col] = ""
            
    # Parallel processing
    results = Parallel(n_jobs=n_jobs)(
        delayed(fetch_from_multiple_jsons)(
            json_files_path=json_dir,
            word=row["word"],
            lemma=row["lemma"],
            max_sentences=max_sentences,
            matching_function=matching_function
        ) for _, row in tqdm(df.iterrows(), total=len(df))
    )

    # Add matched results to DataFrame
    df["matched"] = results

    # Update columns based on "matched" content, keeping original if empty
    df["examples"] = df.apply(
        lambda row: row["matched"].get("sentence") if ast.literal_eval(
            str(row["matched"].get("sentence", ""))) else row["examples"], axis=1
            )
    df["example_urls"] = df.apply(
        lambda row: row["matched"].get("url") if ast.literal_eval(
            str(row["matched"].get("url", ""))) else row["example_urls"], axis=1
            )
    df["doc_id"] = df.apply(
        lambda row: row["matched"].get("doc_id") if ast.literal_eval(
            str(row["matched"].get("doc_id", ""))) else row["doc_id"], axis=1
            )
    df["text_id"] = df.apply(
        lambda row: row["matched"].get("text_id") if ast.literal_eval(
            str(row["matched"].get("text_id", ""))) else row["text_id"], axis=1
            )
    df["categories"] = df.apply(
        lambda row: row["matched"].get("categories") if ast.literal_eval(
            str(row["matched"].get("categories", ""))) else row["categories"], axis=1
            )
    df["domain_type"] = df.apply(
        lambda row: row["matched"].get("domain_type") if ast.literal_eval(
            str(row["matched"].get("domain_type", ""))) else row["domain_type"], axis=1
            )
    df["domain_name"] = df.apply(
        lambda row: row["matched"].get("domain_name") if ast.literal_eval(
            str(row["matched"].get("domain_name", ""))) else row["domain_name"], axis=1
            )

    # Drop "matched" column
    df = df.drop(columns=["matched"])
    return df



# def parallel_processed_dataframe(word_csv_path: str, json_dir: str, max_sentences: int, matching_function: str, n_jobs: int) -> pd.DataFrame:
#     # Read CSV into a pandas DataFrame
#     df = pl.read_csv(word_csv_path).to_pandas()

#     # Parallel processing
#     results = Parallel(n_jobs=n_jobs)(
#         delayed(fetch_from_multiple_jsons)(
#             json_files_path=json_dir,
#             word=row["word"],
#             lemma=row["lemma"],
#             max_sentences=max_sentences,
#             matching_function=matching_function
#         ) for _, row in tqdm(df.iterrows(), total=len(df))
#     )

#     df["matched"] = results
#     df["examples"] = df["matched"].apply(lambda x: x["sentence"])
#     df["example_urls"] = df["matched"].apply(lambda x: x["url"])
#     df["doc_id"] = df["matched"].apply(lambda x: x["doc_id"])
#     df["text_id"] = df["matched"].apply(lambda x: x["text_id"])
#     df["categories"] = df["matched"].apply(lambda x: x["categories"])
#     df["domain_type"] = df["matched"].apply(lambda x: x["domain_type"])
#     df["domain_name"] = df["matched"].apply(lambda x: x["domain_name"])
#     df = df.drop(columns=["matched"])
#     return df

