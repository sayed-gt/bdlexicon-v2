python endpoint.py \
    --input_dir "csv_words/split_csv" \
    --json_files_path "parquet_files" \
    --max_sentences 20 \
    --matching_function "balanced" \
    --output_dir "filtered_sentences_for_words" \
    --n_jobs -1 \
    --parallel
