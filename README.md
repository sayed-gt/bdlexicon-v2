# Bdlexicon Lite
This repository contains the code for fetching sentences for given words. The processed sentence already contains in the `json` format. collected from mongodb corpus using original bdlexicon project.

## How to use

```bash
pip install -r requirements.txt
```

```bash
bash run.sh
```

bash file looks like this:

```bash
python pipeline.py \
    --word_csv_path "bdlexicon_data_chunk_1_words_3640.csv" \
    --json_files_path "sentences_with_meta_all_lemmatized" \
    --max_sentences 20 \
    --matching_function "balanced" \
    --n_jobs 32 \
    --parallel
```
Make sure you pass all of the required arguments.