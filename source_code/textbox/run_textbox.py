# -*-coding:utf-8 -*-

"""
# File       : run_textbox.py
# Time       ：2023/1/29 16:58
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import argparse
from textbox import run_textbox

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='UniLM', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='samsum', help='name of datasets')
    # parser.add_argument('--config_files', type=str, nargs='*', default=["/home/bocheng/source_code/TextBox/textbox/properties/model/transformer.yaml"], help='config files')
    parser.add_argument("--model_path", type=str, default="microsoft/unilm-base-cased")
    args, _ = parser.parse_known_args()

    # run_textbox(model=args.model, dataset=args.dataset, config_file_list=args.config_files, config_dict={})
    run_textbox(model=args.model, dataset=args.dataset, config_dict={"model_path": args.model_path})
