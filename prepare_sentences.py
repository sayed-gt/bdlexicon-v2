"""
This script will read the MongoDB collections, divide text into sentences, and cluster the
sentences into the domains. In a cluster, all sentences will be unique.
"""

import os
import yaml
import json
import argparse

from bdlexicon_utils import (
    check_and_update_sentence_prep_configs, 
    check_dir, 
    get_sentences_from_db, 
    dump_domain_wise_unique_sentences, 
    dump_domain_wise_unique_sentences_and_info, 
    save_configs
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', help='Path to configuration file')

    args = parser.parse_args()
    assert args.config_path.endswith('.yaml'), 'Configuration file must be a YAML file.'

    configs = yaml.safe_load(open(args.config_path))
    configs = check_and_update_sentence_prep_configs(configs)
    check_dir(configs['result_dir'], create=True)
    save_configs(args.config_path, configs['result_dir'], configs)

    if os.path.exists(os.path.join(configs.result_dir, 'meta.json')):
        with open(os.path.join(configs.result_dir, 'meta.json')) as mf:
            meta = json.load(mf)
    else:
        meta = {}

    sentence_data = get_sentences_from_db(configs, meta)
    if configs.include_title:
        dump_domain_wise_unique_sentences_and_info(
            sentence_data, configs.result_dir)
    else:
        dump_domain_wise_unique_sentences(
            sentence_data, configs.result_dir)