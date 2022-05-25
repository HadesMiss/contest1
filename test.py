from transformers import BertTokenizer, BertForMaskedLM
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer
from datasets import Dataset
from parser1 import *
import torch
import json
import numpy as np


class DataMaker:
    def __init__(self, args):
        self.link = []
        self.max_len = args.max_len
        with open('../data/train_triple.jsonl', encoding="utf_8") as file:
            for l in file:
                d = json.loads(l)
                self.link.append([d["subject"], d["object"], d["predicate"], d['salience']])
        self.tokenizer = BertTokenizer.from_pretrained("clue/roberta_chinese_base")
        self.prompt_size = args.prompt_size
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15)
        self.max_len = args.max_len

    def random_masking(self, token_ids):
        """对输入进行随机mask
        """
        rands = np.random.random(len(token_ids))
        source, target = [], []
        for r, t in zip(rands, token_ids):
            if r < 0.15 * 0.8:
                source.append(103)
                target.append(t)
            elif r < 0.15 * 0.9:
                source.append(t)
                target.append(t)
            elif r < 0.15:
                source.append(np.random.choice(self.tokenizer.vocab_size - 1) + 1)
                target.append(t)
            else:
                source.append(t)
                target.append(0)
        return source, target

    def data_maker(self):
        """
        这里挑一个最简单的方法
        s(s_class) [不、很]r t(t_class)
        :return:
        """
        source_list, target_list = [], []

        desc = ['[unused%s]' % i for i in range(1, 1+self.prompt_size*2)]
        desc.insert(self.prompt_size, '[MASK]')
        desc_ids = [self.tokenizer.convert_tokens_to_ids(t) for t in desc]
        pos_id = self.tokenizer.convert_tokens_to_ids(u'很')
        neg_id = self.tokenizer.convert_tokens_to_ids(u'不')

        for s, t, r, label in self.link:
            s_class, r, t_class = r.split('_')
            token_ids = self.tokenizer.encode(s + '(' + s_class + '）' + r + t + '(' + t_class + '）',truncation=True)
            token_ids = token_ids[:1] + desc_ids + token_ids[1:]

            source_ids, target_ids = self.random_masking(token_ids)
            # 长度不满的补充到128
            if label == '0':
                source_ids[self.prompt_size + 1] = 103
                target_ids[self.prompt_size] = neg_id
            elif label == '1':
                source_ids[self.prompt_size] = 103
                target_ids[self.prompt_size] = pos_id
            if len(source_ids) <= self.max_len:
                source_ids = source_ids + [0] * (self.max_len - len(source_ids))
                target_ids = target_ids + [0] * (self.max_len - len(target_ids))
            source_list.append(source_ids)
            target_list.append(target_ids)

        return {'input_ids': torch.LongTensor(source_list),
                'labels': torch.LongTensor(target_list)}


if __name__ == '__main__':
    args = parameter_parser()
    data_maker = DataMaker(args)
    data_train = data_maker.data_maker()
    data_train = Dataset.from_dict(data_train)
    print('    data load done.')

    device = torch.device("cuda:" + str(args.cuda_order) if torch.cuda.is_available() else "cpu")
    print('device:', device)

    model = BertForMaskedLM.from_pretrained(args.model_path)
    print('    No of parameters: ', model.num_parameters())
    print('    model load done.')

    training_args = TrainingArguments(
        output_dir='../data/outputs/',
        overwrite_output_dir=True,
        num_train_epochs=50,
        per_device_train_batch_size=16,
        save_steps=10000,
        do_train=True,
        prediction_loss_only=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_train,
    )
    # Training
    if training_args.do_train:
        checkpoint = None
        train_result = trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload