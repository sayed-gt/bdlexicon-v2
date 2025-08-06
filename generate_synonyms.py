import subprocess
import os
from tqdm import tqdm
import time
import csv
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from typing import List
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser


def concat_df(original_df, output_df):
    # Load CSVs
    df1 = pd.read_csv(original_df)
    df2 = pd.read_csv(output_df)
    # Drop duplicate 'word' entries (keep the last one for each word)
    df1 = df1.drop_duplicates(subset="word", keep="last")
    df2 = df2.drop_duplicates(subset="word", keep="last")
    # Set 'word' column as index
    df1.set_index('word', inplace=True)
    df2.set_index('word', inplace=True)
    # Update df1 with df2 values where index (word) matches
    df1.update(df2)
    # Reset index to make 'word' a column again
    df1.reset_index(inplace=True)
    # Show the first few rows for verification
    print(df1.head())
    # Save back to original file
    df1.to_csv(original_df, index=False)

class WordSchema(BaseModel):
    prompt: str = Field(..., examples=["বাড়ি"])

class SynonymsSchema(BaseModel):
    response: List[str] = Field(..., example=["ঘর", "গৃহ", "বাসস্থান", "আশ্রয়স্থল", "আবাস"])

template = "You are a bengali linguist you will generate 5 brngali synonyms for the given bengali word {word}"
prompt = PromptTemplate(template=template, input_variables=["word"])
output_parser = PydanticOutputParser(pydantic_object=SynonymsSchema)

def main(args):
    dir_name = os.path.dirname(args.save_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)  
    model = ChatGoogleGenerativeAI(
        model=args.model_name, api_key=args.api_key
        ).with_structured_output(SynonymsSchema)
    df = pd.read_csv(args.input_file)
    file_exists = os.path.exists(args.save_path) and os.path.getsize(args.save_path) > 0

    with open(args.save_path, "a", newline="", encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=df.columns)
        
        if not file_exists:
            writer.writeheader()

        for i, row in tqdm(df.iterrows(), total=df.shape[0]):
            new_row = row.to_dict()
            if pd.isna(row["synonyms"]) or str(row["synonyms"]).strip() == "[]":
                try:
                    result = model.invoke(prompt.format(word=row["word"]))
                    new_row["synonyms"] = result.response
                    print(f'Word: {row["word"]}, Synonyms: {result.response}')
                except Exception as e:
                    new_row["synonyms"] = row["synonyms"]
                    print(f"Error processing row {i}: {e}")
                    if "quota" in str(e).lower() or "limit" in str(e).lower():
                        print(f"API quota reached at row {i}. breaking the loop.")
                        break
             
                    continue
                
                time.sleep(args.sleep_time)
            else:
                print("Already Exist! Dumping to csv")

            writer.writerow(new_row)
            

if __name__ == "__main__":
    from typing import NamedTuple
    from dotenv import load_dotenv
    load_dotenv()

    class Args(NamedTuple):
        api_key: str = os.getenv("API_KEY_2")
        model_name: str = "gemini-2.5-flash"
        input_file: str = "data/bangla_words_part_2_repaired.csv"
        save_path: str = "data/bangla_words_part_2_synonym_repaired.csv"
        sleep_time: int = 5

    args = Args()
    main(args)
    concat_df(args.input_file, args.save_path)
    os.remove(args.save_path)

