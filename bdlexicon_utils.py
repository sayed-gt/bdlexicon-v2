import os
import ast
import bkit
import json
import errno
import regex
import hashlib
import traceback
from tqdm import tqdm
import pymongo as mongo
import ruamel.yaml as yaml
from easydict import EasyDict
from datetime import datetime
from typing import List, Tuple, Iterator



def check_and_update_sentence_prep_configs(configs: dict) -> dict:
    # DB configs
    assert 'db_uri' in configs, "'db_uri' is not defined in configs."
    assert 'db_name' in configs, "'db_name' is not defined in configs."

    if 'collections' not in configs:
        configs['collections'] = []
    if 'ignore_collections' not in configs:
        configs['ignore_collections'] = []
    if 'resume' not in configs:
        configs['resume'] = False

    # Data preferences
    if 'include_title' not in configs:
        configs['include_title'] = False
    if 'include_summary' not in configs:
        configs['include_summary'] = False
    if 'include_comments' not in configs:
        configs['include_comments'] = False
    if 'non_bangla_threshold' not in configs:
        configs['non_bangla_threshold'] = 0.000001

    # Data results
    assert 'result_dir' in configs, "'result_dir' is not defined in configs."
    if 'multiple_files' not in configs:
        configs['multiple_files'] = False
    if 'lines_per_file' not in configs:
        configs['lines_per_file'] = 1000000
    if 'dump_duplicates' not in configs:
        configs['dump_duplicates'] = False

    return EasyDict(configs)


def get_data_from_db(configs: EasyDict, meta_info: dict = None):
    if configs.resume:
        assert meta_info is not None

    try:
        db_client = mongo.MongoClient(configs.db_uri)
        db = db_client[configs.db_name]

        db_collections = db.list_collection_names()

        for collection in configs.ignore_collections:
            assert_message = f"{collection} is not a valid collection name"
            assert collection in db_collections, assert_message

        if configs.collections:
            for collection in configs.collections:
                assert_message = f"{collection} is not a valid collection name in {configs.db_name}"
                assert collection in db_collections, assert_message
            db_collections = configs.collections

        ignore_collections = set(configs.ignore_collections)
        db_collections = [
            c for c in db_collections if c not in ignore_collections]

        for i, collection in enumerate(db_collections):
            if configs.resume and collection in meta_info:
                time = meta_info[collection]['last_parse_time']

                time = datetime.fromisoformat(time)
                query = {'parse_time': {'$gt': time}}

                cursor = db[collection].find(query)
                length = db[collection].count_documents(query)
            else:
                cursor = db[collection].find()
                length = db[collection].estimated_document_count()

            loader = tqdm(
                cursor, f"({i+1}/{len(db_collections)}) {collection}", total=length, dynamic_ncols=True)
            for doc in loader:
                data = {'text': doc.get('text', '')}
                data['url'] = doc.get('url', '')

                if 'answer' in doc:
                    data['text'] = doc['answer']

                categories = doc.get('categories', '<UNKNOWN>').split(',')
                data['categories'] = [category.strip().lower()
                                      for category in categories]
                data['domain'] = doc.get('domain', '<UNKNOWN>')

                if configs.include_title:
                    data['title'] = doc.get('title', '')

                    if 'question' in doc:
                        data['title'] = doc['question']

                if configs.include_summary:
                    data['summary'] = doc.get('article_summary', [])

                    if isinstance(data['summary'], str) and data['summary'][0] == '[' and data['summary'][-1] == ']':
                        data['summary'] = ast.literal_eval(data['summary'])

                if configs.include_comments:
                    data['comments'] = doc.get('comments', [])

                    if 'get_suggested_ans' in doc:
                        data['comments'] = doc['get_suggested_ans']

                    if isinstance(data['comments'], str) and data['comments'][0] == '[' and data['comments'][-1] == ']':
                        try:
                            data['comments'] = ast.literal_eval(
                                data['comments'])
                        except:
                            pass

                data['source'] = collection
                data['time'] = doc.get('publish_date', None)
                data['doc_id'] = str(doc.get('_id', None))
                data['domain_name'] = str(doc.get('domain_name', None))
                data['domain_type'] = str(doc.get('domain_type', None))
                data['categories'] = str(doc.get('categories', None))

                yield data
    except Exception:
        print(traceback.format_exc())



def split_into_sentences(text: str) -> List[str]:
    return regex.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![\u0981-\u09fe]\.)(?<=\.|\?|।)\s", text)


def check_text(text: str, non_bangla_threshold: float = 0.3, min_allowed_words: int = 4, max_allowed_words: int = 120) -> bool:
    if not text.strip():
        return False

    words = text.split()
    n_words, n_non_bangla_words = len(words), 0

    if n_words < min_allowed_words or n_words > max_allowed_words:
        return False

    for word in words:
        if non_bangla_threshold < 0.01:
            if not bkit.utils.is_bangla(word):
                n_non_bangla_words += 1
        else:
            if not bkit.utils.contains_bangla(word):
                n_non_bangla_words += 1

    percentage = n_non_bangla_words / n_words
    if percentage >= non_bangla_threshold:
        return False

    return True


def get_sentences_from_db(configs: EasyDict, meta_info: dict = None):
    normalizer = bkit.transform.Normalizer()
    prev_collection = None

    i = 0
    for data in get_data_from_db(configs, meta_info):
        text = ''

        if configs.include_title:
            title = data.get('title', '')
            text += '\n' + title

        doc_text = data.get('text', '')
        text += '\n' + doc_text

        if configs.include_summary:
            summary_list = data.get('summary', [])

            for summary in summary_list:
                text += '\n' + summary

        if configs.include_comments:
            comments = data.get('comments', [])

            for comment in comments:
                text += '\n' + comment

        # post-process text
        text = normalizer(text)
        text = bkit.transform.clean_text(
            text, remove_punctuations=False, remove_digits=False, remove_non_bangla=False)

        sentences = split_into_sentences(text)
        sentences = [s.strip() for s in sentences if check_text(
            s, configs.non_bangla_threshold, 2)]

        result_data = {'sentences': sentences}
        result_data['domain'] = data.get('domain', '<UNKNOWN>')
        result_data['urls'] = [
            data.get('url', None)] * len(result_data['sentences'])
        result_data['doc_ids'] = [
            data.get('doc_id', None)] * len(result_data['sentences'])
        result_data['domain_names'] = [
            data.get('domain_name', None)] * len(result_data['sentences'])
        result_data['domain_types'] = [
            data.get('domain_type', None)] * len(result_data['sentences'])
        result_data['categories'] = [
            data.get('categories', None)] * len(result_data['sentences'])
        result_data['text_ids'] = [hashlib.sha1(
            s.encode('utf-8')).digest().hex() for s in result_data['sentences']]

        if not prev_collection:
            prev_collection = data['source']

        if meta_info is not None:
            collection = data['source']

            if collection in meta_info:
                meta_info[collection]['last_parse_time'] = str(data['time'])
            else:
                meta_info[collection] = {
                    'last_parse_time': str(data['time'])
                }

        if data['source'] != prev_collection:
            with open(os.path.join(configs.result_dir, 'meta.json'), 'w') as mf:
                json.dump(meta_info, mf, ensure_ascii=False, indent=4)
            prev_collection = data['source']

        if (i+1) % 100000 == 0:
            with open(os.path.join(configs.result_dir, 'meta.json'), 'w') as mf:
                json.dump(meta_info, mf, ensure_ascii=False, indent=4)

        i += 1

        yield result_data



def dump_domain_wise_unique_sentences_and_info(
    sentence_data: Tuple[List[dict], Iterator[dict]], dump_dir: str
) -> None:
    domain_sentence_sets = {}
    domain_files = {}

    for data in sentence_data:
        sentences = data["sentences"]
        urls = data["urls"]
        doc_ids = data["doc_ids"]
        text_ids = data["text_ids"]
        categories = data["categories"]
        domain_types = data["domain_types"]
        domain_names = data["domain_names"]

        domain = data["domain"]

        if domain not in domain_files:
            domain_path = os.path.join(dump_dir, f"{domain}.jsonl")
            domain_files[domain] = open(domain_path, "w")

        for i in range(len(sentences)):
            if sentences[i] not in domain_sentence_sets.get(domain, set()):
                temp_data = {
                    "sentence": sentences[i],
                    "url": urls[i],
                    "doc_id": doc_ids[i],
                    "text_id": text_ids[i],
                    "categories": categories[i],
                    "domain_type": domain_types[i],
                    "domain_name": domain_names[i],
                }
                domain_files[domain].write(
                    json.dumps(temp_data, ensure_ascii=False, indent=None)
                )
                domain_files[domain].write("\n")

                domain_sentence_sets.get(domain, set()).add(sentences[i])

    for f in domain_files.values():
        f.close()



def dump_domain_wise_unique_sentences(
    sentence_data: Tuple[List[dict], Iterator[dict]], dump_dir: str
) -> None:
    domain_sentence_sets = {}
    domain_files = {}

    for data in sentence_data:
        sentences = data["sentences"]
        domain = data["domain_type"]

        if domain not in domain_files:
            domain_path = os.path.join(dump_dir, f"{domain}.txt")
            domain_files[domain] = open(domain_path, "w")

        for sentence in sentences:
            if sentence not in domain_sentence_sets.get(domain, set()):
                domain_files[domain].write(f"{sentence}\n")
                domain_sentence_sets.get(domain, set()).add(sentence)

    for f in domain_files.values():
        f.close()



def check_dir(dir_path: str, create: bool = False) -> None:
    if not os.path.exists(dir_path):
        if create:
            os.makedirs(dir_path)
            check_dir(dir_path)
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), dir_path)
    else:
        if not os.path.isdir(dir_path):
            raise NotADirectoryError(
                errno.ENOTDIR, os.strerror(errno.ENOTDIR), dir_path
            )
        

def save_configs(config_file: str, save_dir: str, configs: dict):
    _, ind, bsi = yaml.util.load_yaml_guess_indent(open(config_file))
    yml = yaml.YAML()
    yml.indent(mapping=ind, sequence=ind, offset=bsi)

    dump_file = os.path.join(save_dir, 'configs.yaml')
    dump_file = open(dump_file, 'w')

    yml.dump(dict(configs), dump_file)
