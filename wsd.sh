#!/bin/bash

python wsd.py \
    --word_csv_path "wsd_data/Bangla WSD - 100 wsd.csv" \
    --json_files_path "parquet_sentences" \
    --max_sentences 20 \
    --n_jobs 50
echo "wsd.py executed successfully."

python get_with_csv.py \
    --csv_directory "csv_data" \
    --csv_file_path "wsd_data/Bangla WSD - 100 wsd_repaired.csv" \
    --output_path "wsd_bangla_data.csv"
echo "get_with_csv.py executed successfully."

python get_with_llm.py \
    --word_csv_path "wsd_bangla_data.csv" \
    --api_key "AIzaSyCHz9-F736hmtdZSzr2n8j8HfSxlanXDok"
echo "get_with_llm.py executed successfully."