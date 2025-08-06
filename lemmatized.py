import pandas as pd
import os
import bkit
import json
from bkit.lemmatizer import lemmatize

def lemmatize(data_dir):
    # List all JSON files in the directory
    json_lists = [file for file in os.listdir(data_dir) if file.endswith(".jsonl")]

    for json_list in json_lists:
        # Read JSON file
        df = pd.read_json(os.path.join(data_dir, json_list), lines=True)
        df["lemmatized"] = df["sentence"].apply(lambda x: lemmatize(x))

        data = df.to_dict(orient="records")
        print(data)
        # Write lemmatized JSON file
        with open(os.path.join(data_dir, json_list), "w") as f:
            f.write(json.dumps(data, indent=4, ensure_ascii=False))

# Example usage
data_dir = "temp"
lemmatize(data_dir)

