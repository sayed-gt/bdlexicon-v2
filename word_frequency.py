import json
import os
from elasticsearch import Elasticsearch
import pandas as pd
import csv
from tqdm import tqdm
import re
import unicodedata

def normalize_word(word):
    return unicodedata.normalize("NFC", word.strip())

def doc_frequency_by_word(client, index_name, word):
    response = client.count(
        index=index_name,
        body={
            "query": {
                "match": {
                    "text": word
                }
            }
        }
    )
    return response["count"]



def total_word_frequency_exp(client, index_name, word):
    """Get exact global term frequency in milliseconds."""
    response = client.termvectors(
        index=index_name,
        body={
            "fields": ["text"],
            "term_statistics": True,  # Critical for global counts
        },
        doc={"text": word}  # Dummy doc to trigger analysis
    )
    
    if 'text' in response.get('term_vectors', {}):
        terms = response['term_vectors']['text']['terms']
        return terms.get(word.lower(), {}).get('ttf', 0)  # 'ttf' = total term frequency
    return 0


def iter_docs_containing_word(client, index_name, word, batch_size=1000):
    """Iterate through documents containing the word in text field"""
    response = client.search(
        index=index_name,
        body={
            "query": {
                "match": {"text": word}
            },
            "size": batch_size
        },
        scroll='2m',
    )
    scroll_id = response['_scroll_id']
    hits = response['hits']['hits']

    while hits:
        for hit in hits:
            yield hit['_source']
        
        response = client.scroll(scroll_id=scroll_id, scroll='2m')
        scroll_id = response['_scroll_id']
        hits = response['hits']['hits']


def get_word_frequency(doc, word):
    text = doc.get('text', '')
    pattern = r'\b{}\b'.format(re.escape(word))
    return len(re.findall(pattern, text))


def total_word_frequency(client, index_name, word):
    """Return the total frequency of the word across all documents"""
    total_count = 0
    for doc in iter_docs_containing_word(client, index_name, word):
        total_count += get_word_frequency(doc, word)
    return total_count


# def search_first_occurrence(client, index_name, word):
#     """Return the first occurrence (by publish_date) of the word with URL and source"""
#     response = client.search(
#         index=index_name,
#         body={
#             "query": {
#                 "match": {
#                     "text": word
#                 }
#             },
#             "sort": [
#                 {
#                     "publish_date": {
#                         "order": "asc"
#                     }
#                 }
#             ],
#             "size": 1
#         }
#     )
#     hits = response["hits"]["hits"]
#     if hits:
#         source_doc = hits[0]["_source"]
#         publish_date = source_doc.get("publish_date")
#         url = source_doc.get("url")
#         source = source_doc.get("source")
#         return publish_date, url, source
#     return None, None, None


def search_first_occurrence(client, index_name, word):
    """Return the first valid occurrence (by publish_date) of the word with URL and source"""
    response = client.search(
        index=index_name,
        body={
            "query": {
                "bool": {
                    "must": [
                        {"match": {"text": word}}
                    ],
                    "filter": [
                        {
                            "range": {
                                "publish_date": {
                                    "gte": "1500-01-01",
                                    "lte": "now"
                                }
                            }
                        }
                    ]
                }
            },
            "sort": [
                {"publish_date": {"order": "asc"}}
            ],
            "size": 1
        }
    )

    hits = response["hits"]["hits"]
    if hits:
        source_doc = hits[0]["_source"]
        publish_date = source_doc.get("publish_date")
        url = source_doc.get("url")
        source = source_doc.get("source")
        return publish_date, url, source

    return None, None, None



def main():
    CLIENT_URI = "http://localhost:9201"
    INDEX_NAME = "corpus_collection_index"
    OUTPUT_FILE = "bdlexicon_analysis.csv"
    INPUT_FILE = "bdlexicon_v1.2.csv"

    client = Elasticsearch(CLIENT_URI)
    df = pd.read_csv(INPUT_FILE)

    # Ensure these columns exist in the DF
    for col in ["frequency", "appearance", "source", "url"]:
        if col not in df.columns:
            df[col] = None

    # Detect how many have already been processed in the output (to resume)
    processed_rows = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            processed_rows = max(sum(1 for _ in f) - 1, 0)  # minus header if exists

    # Append if resuming, else write
    mode = "a" if processed_rows > 0 else "w"
    with open(OUTPUT_FILE, mode, newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write header if this is the first time
        if processed_rows == 0:
            writer.writerow(list(df.columns))

        # Iterate through the DF, skipping rows already done
        for i, (_, row) in enumerate(
            tqdm(
                df.iterrows(), total=len(df), initial=processed_rows, desc="Processing words"
            )):
            if i < processed_rows:
                continue

            try:
                if (
                    pd.isna(row["frequency"]) or 
                    pd.isna(row["appearance"]) or 
                    pd.isna(row["source"]) or pd.isna(row["url"])
                    ):
                    word = row["word"].strip()
                    word = normalize_word(word)
                    frequency = total_word_frequency_exp(client, INDEX_NAME, word)
                    tqdm.write(f"Word: {word}, Frequency: {frequency}")
                    appearance, url, source = search_first_occurrence(client, INDEX_NAME, word)

                    # update row
                    row["frequency"] = frequency
                    row["appearance"] = appearance
                    row["source"] = source
                    row["url"] = url

                # Write the (possibly updated) row
                writer.writerow(list(row.values))

            except Exception as e:
                print(f"Error processing word '{row.get('word','')}': {e}")

    client.close()


if __name__ == "__main__":
    main()






# import os
# from elasticsearch import Elasticsearch
# import pandas as pd
# import csv
# from tqdm import tqdm
# import re
# import multiprocessing as mp
# from functools import partial
# import threading
# import time


# def doc_frequency_by_word(client, index_name, word):
#     """Return how many documents contain the word"""
#     response = client.count(
#         index=index_name,
#         body={
#             "query": {
#                 "match": {
#                     "text": word
#                 }
#             }
#         }
#     )
#     return response["count"]
    


# def iter_docs_containing_word(client, index_name, word, batch_size=1000):
#     """Iterate through documents containing the word in text field"""
#     response = client.search(
#         index=index_name,
#         body={
#             "query": {
#                 "match": {"text": word}
#             },
#             "size": batch_size
#         },
#         scroll='2m',
#     )
#     scroll_id = response['_scroll_id']
#     hits = response['hits']['hits']

#     while hits:
#         for hit in hits:
#             yield hit['_source']
        
#         response = client.scroll(scroll_id=scroll_id, scroll='2m')
#         scroll_id = response['_scroll_id']
#         hits = response['hits']['hits']


# def get_word_frequency(doc, word):
#     text = doc.get('text', '')
#     pattern = r'\b{}\b'.format(re.escape(word))
#     return len(re.findall(pattern, text))


# def total_word_frequency(client, index_name, word):
#     """Return the total frequency of the word across all documents"""
#     total_count = 0
#     for doc in iter_docs_containing_word(client, index_name, word):
#         total_count += get_word_frequency(doc, word)
#     return total_count


# def search_first_occurrence(client, index_name, word):
#     """Return the first occurrence (by publish_date) of the word with URL and source"""
#     response = client.search(
#         index=index_name,
#         body={
#             "query": {
#                 "match": {
#                     "text": word
#                 }
#             },
#             "sort": [
#                 {
#                     "publish_date": {
#                         "order": "asc"
#                     }
#                 }
#             ],
#             "size": 1
#         }
#     )
#     hits = response["hits"]["hits"]
#     if hits:
#         source_doc = hits[0]["_source"]
#         publish_date = source_doc.get("publish_date")
#         url = source_doc.get("url")
#         source = source_doc.get("source")
#         return publish_date, url, source
#     return None, None, None


# def process_word_batch(word_batch, client_uri, index_name):
#     """Process a batch of words - this runs in each worker process"""
#     client = Elasticsearch(client_uri)
#     results = []
    
#     for row_data in word_batch:
#         try:
#             word = row_data['word']
            
#             # Check if we need to process this word
#             if (
#                 pd.isna(row_data.get("frequency")) or 
#                 pd.isna(row_data.get("first_appearance")) or 
#                 pd.isna(row_data.get("source")) or 
#                 pd.isna(row_data.get("url"))
#             ):
#                 frequency = total_word_frequency(client, index_name, word)
#                 first_appearance, url, source = search_first_occurrence(client, index_name, word)
                
#                 # Update the row data
#                 row_data["frequency"] = frequency
#                 row_data["first_appearance"] = first_appearance
#                 row_data["source"] = source
#                 row_data["url"] = url
            
#             results.append(row_data)
            
#         except Exception as e:
#             print(f"Error processing word '{row_data.get('word','')}': {e}")
#             results.append(row_data)  # Add the original row even if processing failed
    
#     client.close()
#     return results


# def write_results_safely(results, output_file, columns, lock):
#     """Thread-safe writing of results to CSV"""
#     with lock:
#         with open(output_file, "a", newline="") as csvfile:
#             writer = csv.writer(csvfile)
#             for row_data in results:
#                 row_values = [row_data.get(col) for col in columns]
#                 writer.writerow(row_values)


# def main():
#     CLIENT_URI = "http://localhost:9201"
#     INDEX_NAME = "corpus_collection_index"
#     OUTPUT_FILE = "word_frequencies.csv"
#     INPUT_FILE = "/home/user/Downloads/bdnc_visualization/data/bdlexicon_v1.2.csv"
    
#     # Number of processes to use (adjust as needed)
#     NUM_PROCESSES = mp.cpu_count()
#     BATCH_SIZE = 10  # Number of words per batch
    
#     print(f"Using {NUM_PROCESSES} processes with batch size {BATCH_SIZE}")
    
#     df = pd.read_csv(INPUT_FILE)

#     # Ensure these columns exist in the DF
#     for col in ["frequency", "first_appearance", "source", "url"]:
#         if col not in df.columns:
#             df[col] = None

#     # Detect how many have already been processed
#     processed_rows = 0
#     if os.path.exists(OUTPUT_FILE):
#         with open(OUTPUT_FILE, "r") as f:
#             processed_rows = max(sum(1 for _ in f) - 1, 0)  # minus header if exists

#     # Initialize output file
#     if processed_rows == 0:
#         with open(OUTPUT_FILE, "w", newline="") as csvfile:
#             writer = csv.writer(csvfile)
#             writer.writerow(list(df.columns))

#     # Get rows that need processing
#     rows_to_process = df.iloc[processed_rows:].copy()
    
#     if len(rows_to_process) == 0:
#         print("All rows already processed!")
#         return

#     # Convert rows to dictionaries for easier handling
#     word_data = []
#     for _, row in rows_to_process.iterrows():
#         word_data.append(dict(row))

#     # Create batches
#     batches = []
#     for i in range(0, len(word_data), BATCH_SIZE):
#         batch = word_data[i:i + BATCH_SIZE]
#         batches.append(batch)

#     print(f"Processing {len(word_data)} words in {len(batches)} batches")

#     # Create a lock for thread-safe writing
#     write_lock = threading.Lock()
    
#     # Process batches in parallel
#     with mp.Pool(processes=NUM_PROCESSES) as pool:
#         # Create partial function with fixed arguments
#         process_func = partial(process_word_batch, 
#                              client_uri=CLIENT_URI, 
#                              index_name=INDEX_NAME)
        
#         # Track progress
#         completed_words = 0
#         total_words = len(word_data)
        
#         with tqdm(total=total_words, initial=processed_rows, desc="Processing words") as pbar:
#             # Process batches and write results as they complete
#             for batch_results in pool.imap(process_func, batches):
#                 # Write results immediately to avoid memory buildup
#                 write_results_safely(batch_results, OUTPUT_FILE, df.columns, write_lock)
                
#                 # Update progress
#                 completed_words += len(batch_results)
#                 pbar.update(len(batch_results))
                
#                 # Print some progress info
#                 processed_batch_words = [r['word'] for r in batch_results if r.get('frequency') is not None]
#                 if processed_batch_words:
#                     frequencies = [r.get('frequency', 0) for r in batch_results if r.get('frequency') is not None]
#                     avg_freq = sum(frequencies) / len(frequencies) if frequencies else 0
#                     tqdm.write(f"Batch completed. Avg frequency: {avg_freq:.1f}")

#     print(f"\nProcessing complete! Processed {len(word_data)} words using {NUM_PROCESSES} processes.")


# if __name__ == "__main__":
#     main()