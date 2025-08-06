# import ast
# import glob
# import argparse
# import warnings
# from pipeline import parallel_processed_dataframe, processed_dataframe, timer


# parser = argparse.ArgumentParser()
# parser.add_argument("--word_csv_path", type=str, required=True, help="Path to the CSV file containing the words and their lemmas.")
# parser.add_argument("--json_files_path", type=str, required=True, help="Path to the directory containing the JSON files.")
# parser.add_argument("--max_sentences", type=int, default=20)
# parser.add_argument("--output_path", type=str, default=None)
# parser.add_argument("--matching_function", type=str, default="balanced")
# parser.add_argument("--n_jobs", type=int, default=-1, help="Number of cores to use for parallel processing.")
# parser.add_argument("--parallel", action="store_true")
# args = parser.parse_args()


# @timer
# def main(args):
#     word_csv_path = args.word_csv_path
#     json_files_path = args.json_files_path
#     max_sentences = args.max_sentences
#     output_path = args.output_path
#     matching_function = args.matching_function
#     n_jobs = args.n_jobs

#     if output_path is None:
#         output_path = f"{word_csv_path.removesuffix('.csv')}_repaired.csv"

#     matching_function = args.matching_function
#     if args.parallel:
#         df = parallel_processed_dataframe(word_csv_path, json_files_path, max_sentences, matching_function, n_jobs)
#         df.to_csv(output_path, index=False, encoding="utf-8")
#         return
    
#     df = processed_dataframe(word_csv_path, json_files_path, max_sentences, matching_function)
#     warnings.warn("Parallel processing is not used, this might take a long time")
#     df.to_csv(output_path, index=False, encoding="utf-8")


# if __name__ == "__main__":
#     csv_list = glob.glob("csv_data/split_csv/*.csv")
#     for csv in csv_list:
#         args.word_csv_path = csv
#         main(args)


import ast
import glob
import time
import argparse
import os
import warnings
from pipeline import parallel_processed_dataframe, processed_dataframe, timer


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input CSV files.")
parser.add_argument("--json_files_path", type=str, required=True, help="Path to the directory containing the JSON files.")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save processed CSV files.")
parser.add_argument("--max_sentences", type=int, default=20)
parser.add_argument("--matching_function", type=str, default="balanced")
parser.add_argument("--n_jobs", type=int, default=-1, help="Number of cores to use for parallel processing.")
parser.add_argument("--parallel", action="store_true")
args = parser.parse_args()


@timer
def process_file(input_csv_path, output_csv_path, args):
    if args.parallel:
        df = parallel_processed_dataframe(input_csv_path, args.json_files_path, args.max_sentences, args.matching_function, args.n_jobs)
    else:
        warnings.warn("Parallel processing is not used, this might take a long time")
        df = processed_dataframe(input_csv_path, args.json_files_path, args.max_sentences, args.matching_function)

    df.to_csv(output_csv_path, index=False, encoding="utf-8")


def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    csv_list = glob.glob(os.path.join(input_dir, "*.csv"))
    csv_list = sorted(csv_list, key=lambda x: int(os.path.basename(x).removesuffix('.csv').split('_')[-1]))
    if not csv_list:
        print(f"No CSV files found in {input_dir}")
        return

    for csv_path in csv_list:
        base_name = os.path.basename(csv_path)
        output_csv_path = os.path.join(output_dir, f"{base_name.removesuffix('.csv')}_repaired.csv")
        print(f"Processing {csv_path} -> {output_csv_path}")
        process_file(csv_path, output_csv_path, args)
        os.remove(csv_path)
        time.sleep(10)



if __name__ == "__main__":
    main(args)
