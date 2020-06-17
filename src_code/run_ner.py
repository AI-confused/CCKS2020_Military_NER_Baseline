from __future__ import absolute_import
import argparse
import csv
import logging
import os
import random
import sys
from io import open
import pandas as pd
import numpy as np
import torch
import time
import functools
import collections
import torch.nn as nn
from collections import defaultdict
import gc
import itertools
from multiprocessing import Pool
import functools
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset, Dataset)
from typing import Callable, Dict, List, Generator, Tuple
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from sklearn.metrics import f1_score
import json
import math
from model import BertForNER
from prepare_data import (
  PGD_org,
#   InputExample, 
#   Feature, 
#   collate_fn, 
  load_and_cache_examples,
  set_seed, 
  eval_collate_fn, 
#   pre_process, 
  FGM, 
  Result_whole_doc
)
from itertools import cycle
from torch import optim
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMTokenizer,
    get_linear_schedule_with_warmup,
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForNER, BertTokenizer),
    "albert": (AlbertConfig, AlbertForSequenceClassification, BertTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
}


# print(sys.getrecursionlimit())
sys.setrecursionlimit(8735 * 2080 + 10)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
'''
RESWEIGHTS = [
  'bert.encoder.layer.0.attention.output.resweight',
  'bert.encoder.layer.0.output.resweight',
  'bert.encoder.layer.1.attention.output.resweight',
  'bert.encoder.layer.1.output.resweight',
  'bert.encoder.layer.2.attention.output.resweight',
  'bert.encoder.layer.2.output.resweight',
  'bert.encoder.layer.3.attention.output.resweight',
  'bert.encoder.layer.3.output.resweight',
  'bert.encoder.layer.4.attention.output.resweight',
  'bert.encoder.layer.4.output.resweight',
  'bert.encoder.layer.5.attention.output.resweight',
  'bert.encoder.layer.5.output.resweight',
  'bert.encoder.layer.6.attention.output.resweight',
  'bert.encoder.layer.6.output.resweight',
  'bert.encoder.layer.7.attention.output.resweight',
  'bert.encoder.layer.7.output.resweight',
  'bert.encoder.layer.8.attention.output.resweight',
  'bert.encoder.layer.8.output.resweight',
  'bert.encoder.layer.9.attention.output.resweight',
  'bert.encoder.layer.9.output.resweight',
  'bert.encoder.layer.10.attention.output.resweight',
  'bert.encoder.layer.10.output.resweight',
  'bert.encoder.layer.11.attention.output.resweight',
  'bert.encoder.layer.11.output.resweight',

]
'''


def main():
  parser = argparse.ArgumentParser()

  ## Required parameters
  parser.add_argument("--data_dir", default=None, type=str, required=True,
                      help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
  parser.add_argument("--test_dir", default=None, type=str, required=True)
  parser.add_argument("--stacking", default=None, type=bool, required=True,
                      help="is stacking or not")
  parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                      help="")
  parser.add_argument("--model_type", default=None, type=str, required=True,
                      help="")
  parser.add_argument("--task_type", default=None, type=str, required=True,
                      help="")
  parser.add_argument("--output_dir", default=None, type=str, required=True,
                      help="The output directory where the model predictions and checkpoints will be written.")
  parser.add_argument("--index", default=None, type=int, required=True, help="")
  ## Other parameters
  parser.add_argument("--max_seq_length", default=360, type=int,
                      help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
  parser.add_argument("--num_labels", default=2, type=int)
  parser.add_argument("--max_question_length", default=50, type=int,
                      help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
  parser.add_argument("--overwrite_cache", action='store_true')
  parser.add_argument("--do_lower_case", action='store_true')
  parser.add_argument("--config_name", default=None, type=str)
  parser.add_argument("--tokenizer_name", default=None, type=str)
  parser.add_argument("--do_train", action='store_true',
                      help="Whether to run training.")
  parser.add_argument('--from_tf', action='store_true',
                        help='whether load tensorflow weights')
  parser.add_argument("--FGM", action='store_true',
                      help="Whether to run training.")
  parser.add_argument("--PGD", action='store_true',
                      help="Whether to run training.")
  parser.add_argument("--do_test", action='store_true',
                      help="Whether to run training.")
  parser.add_argument("--do_eval", action='store_true',
                      help="Whether to run eval on the dev set.")
  parser.add_argument("--do_eval_train", action='store_true',
                      help="Whether to run eval on the train set.")
  parser.add_argument("--do_passage_selection", action='store_true',
                      help="Whether to run eval on the train set.")
  parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                      help="Batch size per GPU/CPU for training.")
  parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                      help="Batch size per GPU/CPU for evaluation.")
  parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                      help="Number of updates steps to accumulate before performing a backward/update pass.")
  parser.add_argument("--learning_rate", default=1e-4, type=float,
                      help="The initial learning rate for Adam.")
  parser.add_argument("--k", default=10, type=int, help="")
  parser.add_argument("--weight_decay", default=0.0, type=float,
                      help="Weight deay if we apply some.")
  parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                      help="Epsilon for Adam optimizer.")
  parser.add_argument("--eval_steps", default=-1, type=int,
                      help="")
  parser.add_argument("--train_steps", default=-1, type=int,
                      help="")
  parser.add_argument("--warmup_steps", default=0, type=int,
                      help="Linear warmup over warmup_steps.")
  parser.add_argument('--seed', type=int, default=1,
                      help="random seed for initialization")

  args = parser.parse_args()

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  args.n_gpu = torch.cuda.device_count()

  # Set seed
  set_seed(args)

  args.model_type = args.model_type.lower()
  config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
  config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=args.num_labels)
  tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)

  logger.info("Training/evaluation parameters %s", args)

  args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
                      
  args.label2id={'无标签':0,'试验要素':1,'性能指标':2,'系统组成':3,'任务场景':4}
  args.id2label={0:'无标签',1:'试验要素',2:'性能指标',3:'系统组成',4:'任务场景'}

  if args.do_train:
    # Prepare model
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=args.from_tf, config=config)
    model.to(device)
    
    adv = 'no_adv'
    if args.FGM:
      fgm = FGM(model)
      adv = 'fgm'
      logger.info("***** FGM adv_training *****")
    if args.PGD:
      pgd = PGD_org(model)
      adv = 'pgd'
      logger.info("***** PGD adv_training *****")


    if args.n_gpu > 1:
      model = torch.nn.DataParallel(model)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    
    train_dataset = load_and_cache_examples(args, tokenizer, is_training=1)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size//args.gradient_accumulation_steps)

    num_train_optimization_steps = args.train_steps

    # Prepare optimizer

    param_optimizer = list(model.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
      {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
       'weight_decay': args.weight_decay},
      {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    from transformers import get_linear_schedule_with_warmup
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
#     scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * args.train_steps),
#                                                 num_training_steps=args.train_steps)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=2e-5, max_lr=5e-5,cycle_momentum=False)
    global_step = 0
    logger.info("***** Running training *****")
    logger.info("  Train file: %s", os.path.join(args.data_dir, 'train.csv'))
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)
    logger.info("  Learning rate = %f", args.learning_rate)

    best_acc = 0
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    bar = tqdm(range(num_train_optimization_steps), total=num_train_optimization_steps)
    train_dataloader = cycle(train_dataloader)
    
    output_dir = args.output_dir + "eval_results_{}_{}_{}_{}_{}_{}_{}_{}".format(
                                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                                str(args.max_seq_length),
                                str(args.learning_rate),
                                str(args.train_batch_size),
                                str(args.train_steps),
                                str(args.task_type),
                                adv,
                                str(args.index) + '-fold')
    try:
      os.makedirs(output_dir)
    except:
      pass

    output_eval_file = os.path.join(output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
      writer.write('*' * 80 + '\n')
    for step in bar:
      batch = next(train_dataloader)
      input_ids, input_mask, segment_ids, labels = batch
      loss = model(input_ids=input_ids.to(device), token_type_ids=segment_ids.to(device),
                   attention_mask=input_mask.to(device), labels=labels.to(device))
      if args.n_gpu > 1:
        loss = loss.mean()  # mean() to average on multi-gpu.
      if args.gradient_accumulation_steps > 1:
        loss = loss / args.gradient_accumulation_steps
      tr_loss += loss.item()
      train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
      bar.set_description("loss {}".format(train_loss))
      nb_tr_examples += input_ids.size(0)
      nb_tr_steps += 1

      loss.backward()

      # 对抗训练
      if args.FGM:
        fgm.attack()  # 在embedding上添加对抗扰动
        loss_adv = model(input_ids=input_ids.to(device), token_type_ids=segment_ids.to(device),
                         attention_mask=input_mask.to(device), labels=labels.to(device))
        if args.n_gpu > 1:
          loss_adv = loss_adv.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
          loss_adv = loss_adv / args.gradient_accumulation_steps
        loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        fgm.restore()  # 恢复embedding参数
        
      if args.PGD:
        # pgd 对抗训练
        pgd.backup_grad()
        K = 3  # 超参数
        for t in range(K):
          pgd.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.data
          if t != K - 1:
            model.zero_grad()
          else:
            pgd.restore_grad()
          loss_adv = model(input_ids=input_ids.to(device), token_type_ids=segment_ids.to(device),
                           attention_mask=input_mask.to(device), labels=y_label)
          if args.n_gpu > 1:
            loss_adv = loss_adv.mean()  # mean() to average on multi-gpu.
          if args.gradient_accumulation_steps > 1:
            loss_adv = loss_adv / args.gradient_accumulation_steps
          loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        pgd.restore()  # 恢复embedding参数

      if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1
#         scheduler.step()

      if (step + 1) % (args.eval_steps * args.gradient_accumulation_steps) == 0:
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        logger.info("***** Report result *****")
        logger.info("  %s = %s", 'global_step', str(global_step))
        logger.info("  %s = %s", 'train loss', str(train_loss))

      if args.do_eval and (step + 1) % (args.eval_steps * args.gradient_accumulation_steps) == 0:
        if args.do_eval_train:
            file_list = ['train.csv','dev.csv']
        else:
            file_list = ['dev.csv']
        for file in file_list:
          args.file = file
          eval_dataset = load_and_cache_examples(args, tokenizer, is_training=2)
          eval_sampler = SequentialSampler(eval_dataset)
          eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size//args.gradient_accumulation_steps, collate_fn=eval_collate_fn)

          logger.info("***** Running evaluation *****")
          logger.info("  Eval file = %s", os.path.join(args.data_dir, file))
          logger.info("  Num examples = %d", len(eval_dataset))
          logger.info("  Batch size = %d", args.eval_batch_size)

          model.eval()
          with torch.no_grad():
            result = Result_whole_doc()
            for input_ids, input_mask, segment_ids, examples in tqdm(eval_dataloader):
              input_ids = input_ids.to(device)
              input_mask = input_mask.to(device)
              segment_ids = segment_ids.to(device)
              y_preds = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
              class_preds = y_preds.detach().cpu().numpy()#(p.detach().cpu() for p in y_preds)
              result.update(args, examples, class_preds)

    #           print(result.final_type_strans_pairs)
          scores = result.score()
          model.train()
          result = {'eval_accuracy': scores,
                    'global_step': global_step,
                    'loss': train_loss}

          with open(output_eval_file, "a") as writer:
            writer.write(file + '\n')
            for key in sorted(result.keys()):
              logger.info("  %s = %s", key, str(result[key]))
              writer.write("%s = %s\n" % (key, str(result[key])))
            writer.write('*' * 80)
            writer.write('\n')
          if scores > best_acc and 'dev' in file:
            print("=" * 80)
            print("Best ACC", scores)
            print("Saving Model......")
            best_acc = scores
            # Save a trained model
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(output_dir, "pytorch_model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)
            print("=" * 80)
          else:
            print("=" * 80)
    with open(output_eval_file, "a") as writer:
      writer.write('bert_acc: %f' % best_acc)

  if args.do_test:
    if not args.do_train:
      output_dir = args.output_dir
    else:
      args.do_train = False
    
    eval_dataset = load_and_cache_examples(args, tokenizer, is_training=3)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size//args.gradient_accumulation_steps, collate_fn=eval_collate_fn)

    logger.info("***** Running evaluation *****")
    logger.info("  Eval file = %s", os.path.join(args.test_dir, 'test.csv'))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model = model_class.from_pretrained(output_dir, from_tf=args.from_tf, config=config)
    model.to(device)
    
    if args.n_gpu > 1:
      model = torch.nn.DataParallel(model)
    
    
    model.eval()
    with torch.no_grad():
      result = Result_whole_doc()
      for input_ids, input_mask, segment_ids, examples in tqdm(eval_dataloader):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        y_preds = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        class_preds = y_preds.detach().cpu().numpy()#(p.detach().cpu() for p in y_preds)
        result.update(args, examples, class_preds)

    test_res, test_logit = result.get_test_res()
    with open(os.path.join(args.test_dir, 'submit_sample.json'), 'r', encoding='utf-8') as f:
      json_dict = json.load(f)
      json_keys = list(json_dict.keys())
      for ijson_key, isen_res in zip(json_keys, test_res):
        json_dict[ijson_key] = isen_res

    with open(os.path.join(output_dir, 'test_res.json'), 'w', encoding='utf-8') as f:
      json.dump(json_dict, f, ensure_ascii=False)

    with open(os.path.join(output_dir, 'test_logit_res.json'), 'w', encoding='utf-8') as f:
      json.dump({'res': test_logit}, f, ensure_ascii=False)
    logger.info('test res num - %s',str(len(json_dict)))
    logger.info("writed test res.")


if __name__ == "__main__":
  main()