import pandas as pd
import os

# Load the Parquet file
df = pd.read_csv("csv_words/bangla_word_huge_dataset.csv")

# Directory to save the split CSVs
output_dir = "csv_words/split_csv"
os.makedirs(output_dir, exist_ok=True)

# Number of rows per file
chunk_size = 1000

# Split and save
for i in range(0, len(df), chunk_size):
    chunk = df.iloc[i:i+chunk_size]
    chunk.to_csv(f"{output_dir}/bangla_words_part_{i//chunk_size + 1}.csv", index=False)
