from bkit.lemmatizer import lemmatize
from tqdm import tqdm
import pandas as pd
tqdm.pandas()

df = pd.read_csv("csv_words/bangla_word_huge_dataset.csv")
df["lemma"] = df["word"].progress_apply(lemmatize)
df.to_csv("csv_words/bangla_word_huge_dataset_with_lemma.csv", index=False, encoding="utf-8")