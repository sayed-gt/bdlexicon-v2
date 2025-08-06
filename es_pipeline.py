import os
import ast
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
from elasticsearch import Elasticsearch
from bkit.transform import normalize_punctuation_spaces, clean_text
import re


# Timer decorator
def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper


# Elasticsearch connection setup
def get_elasticsearch_client(host: str, port: int = 9200, username: str = None, password: str = None, use_ssl: bool = False):
    """Create and return Elasticsearch client"""
    scheme = 'https' if use_ssl else 'http'
    node_config = {'host': host, 'port': port, 'scheme': scheme}
    if username and password:
        node_config['http_auth'] = (username, password)
    return Elasticsearch([node_config])


def get_unique_collections(es_client, index_name: str, exclude_patterns=None):
    """Get unique collection names from the _collection field in Elasticsearch"""
    try:
        # Aggregation query to get unique collection names
        agg_query = {
            "aggs": {
                "unique_collections": {
                    "terms": {
                        "field": "_collection.keyword",
                        "size": 10000
                    }
                }
            },
            "size": 0
        }
        
        response = es_client.search(index=index_name, body=agg_query)
        collection_names = [bucket['key'] for bucket in response['aggregations']['unique_collections']['buckets']]
        
        if exclude_patterns:
            filtered_collections = []
            for name in collection_names:
                exclude = False
                for pattern in exclude_patterns:
                    if re.search(pattern, name):
                        exclude = True
                        break
                if not exclude:
                    filtered_collections.append(name)
            return filtered_collections
        
        return collection_names
    except Exception as e:
        print(f"Error getting collection names: {e}")
        return []


def split_text_to_sentences(text: str) -> list:
    """Split text into sentences using common Bengali sentence delimiters"""
    if not text:
        return []
    
    # Clean the text first
    text = text.strip()
    # Split by common Bengali sentence endings
    sentences = re.split(r'[।!?।]', text)
    
    # Clean and filter sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


# Solution 1: Use match query instead of regexp (Recommended)
def balanced_matching_elasticsearch_fixed(word: str, lemma: str, es_client, index_name: str, collection_names: list, max_sentences: int = 1) -> dict:
    """Search for sentences containing word or lemma in Elasticsearch index - Fixed version"""
    word = word.strip()
    lemma = lemma.strip()
    
    result_dict = {
        "sentence": [],
        "url": [],
        "doc_id": [],
        "text_id": [],
        "categories": [],
        "domain_type": [],
        "domain_name": []
    }
    
    # Shuffle collections for random sampling
    shuffled_collections = collection_names.copy()
    random.shuffle(shuffled_collections)
    
    for collection_name in shuffled_collections:
        if len(result_dict["sentence"]) >= max_sentences:
            break
            
        try:
            # Use match query instead of regexp - much more reliable for Unicode text
            query = {
                "bool": {
                    "must": [
                        {
                            "term": {
                                "_collection.keyword": collection_name
                            }
                        },
                        {
                            "bool": {
                                "should": [
                                    {
                                        "match": {
                                            "text": {
                                                "query": word,
                                                "operator": "and"
                                            }
                                        }
                                    },
                                    {
                                        "match": {
                                            "text": {
                                                "query": lemma,
                                                "operator": "and"
                                            }
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                }
            }
            
            # Execute search
            search_body = {
                "query": query,
                "size": max_sentences - len(result_dict["sentence"])
            }
            
            response = es_client.search(index=index_name, body=search_body)
            
            for hit in response['hits']['hits']:
                if len(result_dict["sentence"]) >= max_sentences:
                    break
                
                doc = hit['_source']
                
                # Split text into sentences
                sentences = split_text_to_sentences(doc.get("text", ""))
                
                # Find sentences containing the word or lemma (case-insensitive)
                matching_sentences = []
                for sentence in sentences:
                    if (word.lower() in sentence.lower() or 
                        lemma.lower() in sentence.lower()):
                        matching_sentences.append(sentence)
                
                # Add matching sentences to result
                for sentence in matching_sentences:
                    if len(result_dict["sentence"]) >= max_sentences:
                        break
                    
                    result_dict["sentence"].append(sentence)
                    result_dict["url"].append(doc.get("url", ""))
                    result_dict["doc_id"].append(hit['_id'])
                    result_dict["text_id"].append(doc.get("text_hash", ""))
                    result_dict["categories"].append(doc.get("categories", ""))
                    result_dict["domain_type"].append(doc.get("source_type", ""))
                    result_dict["domain_name"].append(doc.get("domain", ""))
                    
        except Exception as e:
            print(f"Error processing collection {collection_name}: {e}")
            continue
    
    return result_dict


# Solution 2: Use wildcard query (Alternative approach)
def balanced_matching_elasticsearch_wildcard(word: str, lemma: str, es_client, index_name: str, collection_names: list, max_sentences: int = 1) -> dict:
    """Search using wildcard queries - works better with Unicode"""
    word = word.strip()
    lemma = lemma.strip()
    
    result_dict = {
        "sentence": [],
        "url": [],
        "doc_id": [],
        "text_id": [],
        "categories": [],
        "domain_type": [],
        "domain_name": []
    }
    
    shuffled_collections = collection_names.copy()
    random.shuffle(shuffled_collections)
    
    for collection_name in shuffled_collections:
        if len(result_dict["sentence"]) >= max_sentences:
            break
            
        try:
            # Use wildcard query instead of regexp
            query = {
                "bool": {
                    "must": [
                        {
                            "term": {
                                "_collection.keyword": collection_name
                            }
                        },
                        {
                            "bool": {
                                "should": [
                                    {
                                        "wildcard": {
                                            "text": f"*{word}*"
                                        }
                                    },
                                    {
                                        "wildcard": {
                                            "text": f"*{lemma}*"
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                }
            }
            
            search_body = {
                "query": query,
                "size": max_sentences - len(result_dict["sentence"])
            }
            
            response = es_client.search(index=index_name, body=search_body)
            
            for hit in response['hits']['hits']:
                if len(result_dict["sentence"]) >= max_sentences:
                    break
                
                doc = hit['_source']
                sentences = split_text_to_sentences(doc.get("text", ""))
                
                matching_sentences = []
                for sentence in sentences:
                    if (word.lower() in sentence.lower() or 
                        lemma.lower() in sentence.lower()):
                        matching_sentences.append(sentence)
                
                for sentence in matching_sentences:
                    if len(result_dict["sentence"]) >= max_sentences:
                        break
                    
                    result_dict["sentence"].append(sentence)
                    result_dict["url"].append(doc.get("url", ""))
                    result_dict["doc_id"].append(hit['_id'])
                    result_dict["text_id"].append(doc.get("text_hash", ""))
                    result_dict["categories"].append(doc.get("categories", ""))
                    result_dict["domain_type"].append(doc.get("source_type", ""))
                    result_dict["domain_name"].append(doc.get("domain", ""))
                    
        except Exception as e:
            print(f"Error processing collection {collection_name}: {e}")
            continue
    
    return result_dict


# Solution 3: Fixed regex version (if you must use regex)
def safe_regex_escape(text: str) -> str:
    """Safely escape text for Elasticsearch regex, handling Unicode properly"""
    # List of regex special characters that need escaping
    special_chars = r'\.^$*+?{}[]|()'
    escaped = ""
    
    for char in text:
        if char in special_chars:
            escaped += "\\" + char
        else:
            escaped += char
    
    return escaped


def balanced_matching_elasticsearch_safe_regex(word: str, lemma: str, es_client, index_name: str, collection_names: list, max_sentences: int = 1) -> dict:
    """Search using properly escaped regex for Unicode text"""
    word = word.strip()
    lemma = lemma.strip()
    
    result_dict = {
        "sentence": [],
        "url": [],
        "doc_id": [],
        "text_id": [],
        "categories": [],
        "domain_type": [],
        "domain_name": []
    }
    
    shuffled_collections = collection_names.copy()
    random.shuffle(shuffled_collections)
    
    for collection_name in shuffled_collections:
        if len(result_dict["sentence"]) >= max_sentences:
            break
            
        try:
            # Use safe regex escaping without \b word boundaries
            escaped_word = safe_regex_escape(word)
            escaped_lemma = safe_regex_escape(lemma)
            
            query = {
                "bool": {
                    "must": [
                        {
                            "term": {
                                "_collection.keyword": collection_name
                            }
                        },
                        {
                            "bool": {
                                "should": [
                                    {
                                        "regexp": {
                                            "text": f".*{escaped_word}.*"
                                        }
                                    },
                                    {
                                        "regexp": {
                                            "text": f".*{escaped_lemma}.*"
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                }
            }
            
            search_body = {
                "query": query,
                "size": max_sentences - len(result_dict["sentence"])
            }
            
            response = es_client.search(index=index_name, body=search_body)
            
            for hit in response['hits']['hits']:
                if len(result_dict["sentence"]) >= max_sentences:
                    break
                
                doc = hit['_source']
                sentences = split_text_to_sentences(doc.get("text", ""))
                
                matching_sentences = []
                for sentence in sentences:
                    if (word.lower() in sentence.lower() or 
                        lemma.lower() in sentence.lower()):
                        matching_sentences.append(sentence)
                
                for sentence in matching_sentences:
                    if len(result_dict["sentence"]) >= max_sentences:
                        break
                    
                    result_dict["sentence"].append(sentence)
                    result_dict["url"].append(doc.get("url", ""))
                    result_dict["doc_id"].append(hit['_id'])
                    result_dict["text_id"].append(doc.get("text_hash", ""))
                    result_dict["categories"].append(doc.get("categories", ""))
                    result_dict["domain_type"].append(doc.get("source_type", ""))
                    result_dict["domain_name"].append(doc.get("domain", ""))
                    
        except Exception as e:
            print(f"Error processing collection {collection_name}: {e}")
            continue
    
    return result_dict


# Updated unified function with match query
def unified_matching_elasticsearch_fixed(word: str, lemma: str, es_client, index_name: str, collection_names: list, max_sentences: int = 1) -> dict:
    """Unified matching function using match query - most reliable approach"""
    word = word.strip()
    lemma = lemma.strip()
    
    result_dict = {
        "sentence": [],
        "url": [],
        "doc_id": [],
        "text_id": [],
        "categories": [],
        "domain_type": [],
        "domain_name": []
    }
    
    try:
        # Single query across all collections using match query
        query = {
            "bool": {
                "must": [
                    {
                        "terms": {
                            "_collection.keyword": collection_names
                        }
                    },
                    {
                        "bool": {
                            "should": [
                                {
                                    "match": {
                                        "text": {
                                            "query": word,
                                            "operator": "and"
                                        }
                                    }
                                },
                                {
                                    "match": {
                                        "text": {
                                            "query": lemma,
                                            "operator": "and"
                                        }
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        }
        
        # Execute search with random sorting
        search_body = {
            "query": query,
            "size": max_sentences * 3,  # Get more results for better selection
            "sort": [
                {
                    "_script": {
                        "type": "number",
                        "script": {
                            "source": "Math.random()"
                        }
                    }
                }
            ]
        }
        
        response = es_client.search(index=index_name, body=search_body)
        
        for hit in response['hits']['hits']:
            if len(result_dict["sentence"]) >= max_sentences:
                break
            
            doc = hit['_source']
            sentences = split_text_to_sentences(doc.get("text", ""))
            
            matching_sentences = []
            for sentence in sentences:
                if (word.lower() in sentence.lower() or 
                    lemma.lower() in sentence.lower()):
                    matching_sentences.append(sentence)
            
            for sentence in matching_sentences:
                if len(result_dict["sentence"]) >= max_sentences:
                    break
                
                result_dict["sentence"].append(sentence)
                result_dict["url"].append(doc.get("url", ""))
                result_dict["doc_id"].append(hit['_id'])
                result_dict["text_id"].append(doc.get("text_hash", ""))
                result_dict["categories"].append(doc.get("categories", ""))
                result_dict["domain_type"].append(doc.get("source_type", ""))
                result_dict["domain_name"].append(doc.get("domain", ""))
                
    except Exception as e:
        print(f"Error in unified matching: {e}")
    
    return result_dict


def fetch_from_elasticsearch_fixed(es_client, index_name: str, collection_names: list, word: str, lemma: str, max_sentences: int, matching_function: str) -> dict:
    """Main function to fetch sentences from Elasticsearch index - Fixed version"""
    
    if matching_function == "unified":
        matching_func = unified_matching_elasticsearch_fixed
    elif matching_function == "wildcard":
        matching_func = balanced_matching_elasticsearch_wildcard
    elif matching_function == "safe_regex":
        matching_func = balanced_matching_elasticsearch_safe_regex
    else:  # Default to match query approach
        matching_func = balanced_matching_elasticsearch_fixed
    
    result_dict = matching_func(word, lemma, es_client, index_name, collection_names, max_sentences)
    
    # Apply text normalization
    normalized_sentences = []
    for sentence in result_dict["sentence"]:
        normalized = normalize_punctuation_spaces(sentence)
        cleaned = clean_text(normalized) + " ।"
        normalized_sentences.append(cleaned)
    
    result_dict["sentence"] = normalized_sentences
    return result_dict


@timer
def processed_dataframe_elasticsearch_fixed(word_csv_path: str, es_host: str, es_port: int, index_name: str, 
                                          max_sentences: int, matching_function: str = "match", es_username: str = None, 
                                          es_password: str = None, exclude_collections=None) -> pd.DataFrame:
    """Process DataFrame using Elasticsearch as data source - Fixed version"""
    
    # Read the word CSV
    df = pl.read_csv(word_csv_path).to_pandas()
    
    # Setup Elasticsearch connection
    es_client = get_elasticsearch_client(es_host, es_port, es_username, es_password)
    
    # Get collection names
    collection_names = get_unique_collections(es_client, index_name, exclude_collections)
    print(f"Found {len(collection_names)} collections to search")
    
    # Initialize results list
    matched_sentences = []
    
    # Process each word/lemma pair
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing words"):
        result = fetch_from_elasticsearch_fixed(
            es_client=es_client,
            index_name=index_name,
            collection_names=collection_names,
            word=row["word"], 
            lemma=row["lemma"],
            max_sentences=max_sentences,
            matching_function=matching_function
        )
        matched_sentences.append(result)
    
    # Add results to DataFrame
    df["matched"] = matched_sentences
    df["examples"] = df["matched"].apply(lambda x: x["sentence"])
    df["example_urls"] = df["matched"].apply(lambda x: x["url"])
    df["doc_id"] = df["matched"].apply(lambda x: x["doc_id"])
    df["text_id"] = df["matched"].apply(lambda x: x["text_id"])
    df["categories"] = df["matched"].apply(lambda x: x["categories"])
    df["domain_type"] = df["matched"].apply(lambda x: x["domain_type"])
    df["domain_name"] = df["matched"].apply(lambda x: x["domain_name"])
    
    # Clean up
    df = df.drop(columns=["matched"])
    
    return df


@timer
def parallel_processed_dataframe_elasticsearch_fixed(word_csv_path: str, es_host: str, es_port: int, index_name: str,
                                                   max_sentences: int, matching_function: str = "match", n_jobs: int = 4, 
                                                   es_username: str = None, es_password: str = None, 
                                                   exclude_collections=None) -> pd.DataFrame:
    """Parallel processing version using Elasticsearch - Fixed version"""
    
    df = pd.read_csv(word_csv_path)
    
    # Ensure required columns exist
    for col in ["examples", "example_urls", "doc_id", "text_id", "categories", "domain_type", "domain_name"]:
        if col not in df.columns:
            df[col] = ""
    
    # Setup Elasticsearch connection to get collection names
    es_client = get_elasticsearch_client(es_host, es_port, es_username, es_password)
    collection_names = get_unique_collections(es_client, index_name, exclude_collections)
    
    print(f"Found {len(collection_names)} collections to search")
    print(f"Using matching function: {matching_function}")
    
    # Parallel processing function
    def process_row(row_data):
        es_client_local = get_elasticsearch_client(es_host, es_port, es_username, es_password)
        
        result = fetch_from_elasticsearch_fixed(
            es_client=es_client_local,
            index_name=index_name,
            collection_names=collection_names,
            word=row_data["word"],
            lemma=row_data["lemma"],
            max_sentences=max_sentences,
            matching_function=matching_function
        )
        
        return result
    
    # Execute parallel processing
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_row)(row.to_dict()) 
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing words in parallel")
    )
    
    # Add matched results to DataFrame
    df["matched"] = results
    
    # Update columns based on matched content
    df["examples"] = df.apply(
        lambda row: row["matched"].get("sentence", []) if row["matched"].get("sentence") else row["examples"], 
        axis=1
    )
    df["example_urls"] = df.apply(
        lambda row: row["matched"].get("url", []) if row["matched"].get("url") else row["example_urls"], 
        axis=1
    )
    df["doc_id"] = df.apply(
        lambda row: row["matched"].get("doc_id", []) if row["matched"].get("doc_id") else row["doc_id"], 
        axis=1
    )
    df["text_id"] = df.apply(
        lambda row: row["matched"].get("text_id", []) if row["matched"].get("text_id") else row["text_id"], 
        axis=1
    )
    df["categories"] = df.apply(
        lambda row: row["matched"].get("categories", []) if row["matched"].get("categories") else row["categories"], 
        axis=1
    )
    df["domain_type"] = df.apply(
        lambda row: row["matched"].get("domain_type", []) if row["matched"].get("domain_type") else row["domain_type"], 
        axis=1
    )
    df["domain_name"] = df.apply(
        lambda row: row["matched"].get("domain_name", []) if row["matched"].get("domain_name") else row["domain_name"], 
        axis=1
    )
    
    # Drop the matched column
    df = df.drop(columns=["matched"])
    return df


# Updated example usage
if __name__ == "__main__":
    # Configuration
    ES_HOST = "localhost"
    ES_PORT = 9200
    INDEX_NAME = "corpus_collection_index"
    WORD_CSV_PATH = "csv_data/bangla_words_temp.csv"
    MAX_SENTENCES = 20
    MATCHING_FUNCTION = "match"  # Options: "match", "unified", "wildcard", "safe_regex"
    N_JOBS = 20
    ES_USERNAME = None
    ES_PASSWORD = None
    EXCLUDE_COLLECTIONS = ["system.*", "config.*"]
        
    parallel_result_df = parallel_processed_dataframe_elasticsearch_fixed(
        word_csv_path=WORD_CSV_PATH,
        es_host=ES_HOST,
        es_port=ES_PORT,
        index_name=INDEX_NAME,
        max_sentences=MAX_SENTENCES,
        matching_function=MATCHING_FUNCTION,
        n_jobs=N_JOBS,
        exclude_collections=EXCLUDE_COLLECTIONS
    )
    
    parallel_result_df.to_csv("elasticsearch_results.csv", index=False, encoding="utf-8")