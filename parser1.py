#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/3 下午9:03
# @Author  : liu yuhan
# @FileName: parser.py
# @Software: PyCharm

import argparse


def parameter_parser():
    parser = argparse.ArgumentParser(description='for cnc ner')
    # model parameter
    parser.add_argument("--model_path", type=str, default="albert-base-v2")
    parser.add_argument("--pretrain_model_path", type=str, default="../roberta_chinese")
    parser.add_argument("--model_save_path", type=int, default=101)
    parser.add_argument("--eval_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument("--cuda_order", type=str, default='0')
    parser.add_argument("--max_len", type=int, default=48)
    # file
    parser.add_argument("--train_file_path", type=str, default="../train.tsv")
    parser.add_argument("--test_file_path", type=str, default="../test.tsv")
    # predict parameter
    parser.add_argument("--best_epoch_without_feature", type=int, default=10)
    parser.add_argument("--best_epoch_with_feature", type=int, default=3)
    # ptunning parameter
    parser.add_argument("--prompt_size", type=int, default=3)

    return parser.parse_args()
