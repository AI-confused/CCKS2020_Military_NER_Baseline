import json
from pathlib import Path
import time
import torch
import numpy as np
import random
import pandas as pd
import re
import itertools
from tqdm import tqdm
from pandas.io.json._json import JsonReader
from typing import Callable, Dict, List, Generator, Tuple
from multiprocessing import Pool
import os
import operator
import logging
import functools
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, Dataset
from collections import defaultdict
from rouge import Rouge
from elasticsearch import Elasticsearch
import json

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)




class InputExample(object):
  """
  一个（问题-文档）样本，待拆分
  """

  def __init__(self, guid, text_a, text_b, start=None, end=None,label_type=None, str_ans=None):
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.start = start
    self.end = end
    self.label_type=label_type
    self.str_ans = str_ans


class Feature(object):
  """
  一个（问题-拆分后的文档片段）训练-测试样本，用于输入模型
  """
  def __init__(
          self,
          example_id,
#           doc_start,  # 训练样本中文档片段在原始文档中的起点位置
#           question_len,
          tokenized_to_original_index,
          input_tokens,
          input_ids,
          input_mask,
          segment_ids,
#           question=None,
#           bert_start_position=None,  # 输入Bert模型的序列中的start标签
#           bert_end_position=None,  # 输入Bert模型的序列中的end标签
          label=None,
          passage=None,
#           origin_start=None,  # 原始文档答案起点
#           origin_end=None,  # 原始文档答案终点
#           token_start=None,  # token后的文档答案起点
#           token_end=None,  # token后的文档答案终点
#           doc_end=None,  # 训练样本中文档片段在原始文档中的终点位置
#           doc_id=None,
#           bert_doc_end=None,  # 在bert的输入序列中，padding之前有效文档片段的结尾索引（闭区间）
#           bert_answer_span=None,  # 输入模型的序列中答案标签的长度
#           doc_length=None,  # 输入模型的序列中文档片段长度
#           cls_logit=None,
#           best_scores=None,
#           result=None,  # 答案的span
#           new_score=None,  # 用于对答案span进行排序的分数
#           search_pred=None,
          str_ans=None,
          label_id=None
  ):
    self.example_id = example_id
#     self.doc_id = doc_id
    self.passage = passage
#     self.doc_start = doc_start
#     self.doc_end = doc_end
#     self.doc_length = doc_length
#     self.bert_doc_end = bert_doc_end
#     self.question_len = question_len
    self.tokenized_to_original_index = tokenized_to_original_index
    self.input_tokens = input_tokens
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
#     self.question = question
#     self.bert_start_position = bert_start_position
#     self.bert_end_position = bert_end_position
#     self.bert_answer_span = bert_answer_span
    self.label = label
#     self.origin_start = origin_start
#     self.origin_end = origin_end
#     self.token_start = token_start
#     self.token_end = token_end
#     self.cls_logit = cls_logit
#     self.best_scores = best_scores
#     self.result = result
#     self.new_score = new_score
#     self.search_pred = search_pred
    self.str_ans = str_ans
    self.label_id=label_id # 用作训练用的label

    
    
def load_and_cache_examples(args, tokenizer, is_training):
  # Load data features from cache or dataset file
  if is_training==1:
    cached_features_file = os.path.join(
        args.data_dir,
        "cached-{}-{}-{}-{}".format(
            "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(args.task_type),
            ),
    )   
  elif is_training==2:
    cached_features_file = os.path.join(
        args.data_dir,
        "cached-{}-{}-{}-{}-{}".format(
            "dev",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            args.file.replace('.', '_'),
            str(args.task_type),
            ),
    )
  else:
    cached_features_file = os.path.join(
        args.data_dir,
        "cached-{}-{}-{}-{}".format(
            "test",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(args.task_type),
            ),
    )

  if os.path.exists(cached_features_file) and not args.overwrite_cache:
    logger.info("Loading features from cached file %s", cached_features_file)
    features = torch.load(cached_features_file)
  else:
    logger.info("Creating features from dataset file at %s", args.data_dir)
    if is_training==1:
      examples = read_examples(os.path.join(args.data_dir, 'train.csv'), is_training)
    elif is_training==2:
      examples = read_examples(os.path.join(args.data_dir, args.file), is_training)
    else:
      examples = read_examples(os.path.join(args.test_dir, 'test.csv'), is_training)
      
    convert_func = functools.partial(convert_examples_to_features,
                                     tokenizer=tokenizer,
                                     max_seq_length=args.max_seq_length,
                                     max_question_length=args.max_question_length,
                                     is_training=is_training,
                                     label_map=args.label2id)
    logger.info('generating features...')
    with Pool(10) as p:
      features = p.map(convert_func, examples)
    logger.info("Saving features into cached file %s", cached_features_file)
    torch.save(features, cached_features_file)

  # Convert to Tensors and build dataset
  if is_training == 1:
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label_id for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
  elif is_training in [2, 3]:
    dataset = TextDataset(features)
  return dataset


def read_examples(input_file, is_training):
  """
  从数据集文件读取数据生成样本 InputExample
  """
  df = pd.read_csv(input_file)
  logger.info('reading examples...')
  # train
  if is_training == 1:
    examples = []
    for i_index, val in enumerate(df[['org_text', 'label_type', 'start_pos', 'end_pos', 'label_text']].values):
      assert type(eval(val[3]))==list,'not list type'
      examples.append(InputExample(
        guid=i_index,
        text_a=val[0],
        text_b=None,
        start=eval(val[2]),
        end=eval(val[3]),
        label_type=eval(val[1]),
        str_ans=eval(val[-1])
      ))
      assert len(eval(val[2])) == len(eval(val[3]))
  # eval
  elif is_training == 2:
    examples = []
    for i_index, val in enumerate(df[['org_text', 'label_type', 'start_pos', 'end_pos', 'label_text']].values):
      examples.append(InputExample(
        guid=i_index,
        text_a=val[0],
        text_b=None,
        start=eval(val[2]),
        end=eval(val[3]),
        label_type=eval(val[1]),
        str_ans=eval(val[-1])
      ))
      assert len(eval(val[2])) == len(eval(val[3]))
  # test
  else:
    examples = []
    for i_index, val in enumerate(df[['org_text']].values):
      examples.append(InputExample(
        guid=i_index,
        text_a=val[0],
        text_b=None,
      ))
  return examples


def convert_examples_to_features(example, tokenizer, max_seq_length, max_question_length, is_training, label_map):
  """
  InputExample处理为可以输入模型的Feature
  """
  passage_words = list(example.text_a)
  original_to_tokenized_index = []
  tokenized_to_original_index = []
  passage_tokens = []

  # 构建原始文档和token文档的互相映射，用于查找索引答案
  for i, word in enumerate(passage_words):
    original_to_tokenized_index.append(len(passage_tokens))
    sub_tokens = tokenizer.tokenize(word)
    for sub_token in sub_tokens:
      tokenized_to_original_index.append(i)
      passage_tokens.append(sub_token)
  assert len(tokenized_to_original_index) == len(passage_tokens)

  max_doc_length = max_seq_length - 2
  _truncate_seq_pair(passage_tokens, '', max_doc_length)
  assert len(passage_tokens) <= max_doc_length

  # train or dev，把原始文本中的各个实体的起始位置标签映射到tokenize后的文本中
  if is_training in [1,2]:
    new_starts, new_ends = [], []
    for istart, iend in zip(example.start, example.end):
      new_starts.append(original_to_tokenized_index[istart])
      new_ends.append(original_to_tokenized_index[iend])

    # 按照升序重新排列实体(s, e)
    label_tuple = []
    for istart, iend, itype, ians in zip(new_starts, new_ends, example.label_type, example.str_ans):
      label_tuple.append((istart, iend, label_map[itype], ians))
    assert len(label_tuple) == len(example.start)
    label_tuple = sorted(label_tuple, key=lambda x: x[0])

    # 给序列添加对应的实体label
    example_label = [0] # ['cls']
    entity_index = 0
    s, e = label_tuple[entity_index][0], label_tuple[entity_index][1]
    for i, token in enumerate(passage_tokens):
      if i < s:
        example_label.append(0) # label-'O'
      elif i==s:
        example_label.append(label_tuple[entity_index][2]) # label-'B'
        # 实体只有一个token长度
        if s==e:
          entity_index += 1
          if entity_index == len(label_tuple):
            s = len(passage_tokens)
          else:
            s, e = label_tuple[entity_index][0], label_tuple[entity_index][1] # 双指针移动到下一个实体的(s,e)
      elif s<i<e:
        example_label.append(label_tuple[entity_index][2]+4) # label-'I'
      elif i==e:
        example_label.append(label_tuple[entity_index][2]+4) # label-'I'
        entity_index += 1
        if entity_index == len(label_tuple):
          s = len(passage_tokens)
        else:
          s, e = label_tuple[entity_index][0], label_tuple[entity_index][1] # 双指针移动到下一个实体的(s,e)
    example_label.append(0) # ['sep']
    if len(example_label) != len(passage_tokens) + 2:
      print('error: ', len(example_label), len(passage_tokens), example_label, passage_tokens, label_tuple)
    assert len(passage_tokens)+2 == len(example_label)
  
  # 生成Feature
  input_tokens = ['[CLS]'] + passage_tokens + ['[SEP]']
  if is_training != 3:
    assert len(input_tokens) <= max_seq_length and len(input_tokens) == len(example_label)
  else:
    assert len(input_tokens) <= max_seq_length
  segment_ids = [0] * len(input_tokens)
  input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
  input_mask = [1] * len(input_ids)
  # padding
  padding_length = max_seq_length - len(input_ids)
  input_ids += ([0] * padding_length)
  input_mask += ([0] * padding_length)
  segment_ids += ([0] * padding_length)
  assert len(input_ids) == max_seq_length
    
  # train & dev
  if is_training in [1,2]:
    example_label += ([0] * padding_length)
    feature = Feature(
          example_id=example.guid,
          passage=example.text_a,
          tokenized_to_original_index=tokenized_to_original_index,
          input_tokens=input_tokens,
          input_ids=input_ids,
          input_mask=input_mask,
          segment_ids=segment_ids,
          label_id=example_label, # for train
          label=label_tuple, # for eval
    )
  # test
  else:
    feature = Feature(
          example_id=example.guid,
          passage=example.text_a,
          tokenized_to_original_index=tokenized_to_original_index,
          input_tokens=input_tokens,
          input_ids=input_ids,
          input_mask=input_mask,
          segment_ids=segment_ids,
    )

  return feature

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()



class TextDataset(Dataset):
  def __init__(self, examples: List[Feature]):
    self.examples = examples

  def __len__(self) -> int:
    return len(self.examples)

  def __getitem__(self, index):
    return self.examples[index]


def collate_fn(examples: List[Feature]):
  """
  对训练数据集的一个batch处理为tensor
  """
  input_ids = torch.stack([torch.tensor(example[0].input_ids, dtype=torch.long) for example in examples], 0)
  input_mask = torch.stack([torch.tensor(example[0].input_mask, dtype=torch.long) for example in examples], 0)
  segment_ids = torch.stack([torch.tensor(example[0].segment_ids, dtype=torch.long) for example in examples], 0)
  labels= torch.stack([torch.tensor(example[0].label_id, dtype=torch.long) for example in examples], 0)


  return [input_ids, input_mask, segment_ids, labels]


def eval_collate_fn(examples: List[Feature]):
  """
  对验证数据集的一个batch处理为tensor
  """
  input_ids = torch.stack([torch.tensor(example.input_ids, dtype=torch.long) for example in examples], 0)
  input_mask = torch.stack([torch.tensor(example.input_mask, dtype=torch.long) for example in examples], 0)
  segment_ids = torch.stack([torch.tensor(example.segment_ids, dtype=torch.long) for example in examples], 0)

  return input_ids, input_mask, segment_ids, examples


class Result_whole_doc(object):
  """
  存储并且计算最终答案分数的类
  """
  def __init__(self):
    self.final_type_strans_pairs=[]
    self.ground_truth_strans_pairs=[]
    self.from_truelabelid_round_truth_strans_pairs=[]
    self.test_res=[]
    self.test_logit=[]
    self.test_id=10000
    
  # 存储每个batch的预测结果
  def update(self, 
             args,
             examples,
             class_preds):
    pre_labels = np.argmax(class_preds, axis=2)
    for example, ipre_seq_label in zip(examples, pre_labels):
      #单个样本信息
      token2orgindex = example.tokenized_to_original_index
      org_text = example.passage
      token_len = len(example.input_tokens) # 非padding部分长度
      #保存字典格式的文件 for test
      self.test_logit.append({self.test_id: {
                                'token2orgindex': token2orgindex, 
                                'org_text': org_text,
                                'token_len': token_len, 
                                }})
      self.get_a_sentence_label_v2(self.final_type_strans_pairs, ipre_seq_label, token_len, org_text, token2orgindex, args.id2label)
      self.test_id+=1
    if args.do_train:
      for iexample in examples:
        for istart, iend, ilabel_type, istr_ans in iexample.label:
          self.ground_truth_strans_pairs.append(args.id2label[ilabel_type] + '_' + istr_ans)
#     print(len(self.ground_truth_strans_pairs), len(self.testq_logit))
    
    
#   def change_ilabel(self,label_list):
#     for i_index,ilabel in enumerate(label_list):
#       if ilabel>4:
#         label_list[i_index]-=4
#     return label_list

  
  def get_sen_label_helper(self, predict_label, end_label, ipos, ipre_seq_label, id2label, token2orgindex, a_senten_res,
                           save_list,org_text):
    #predict_label- begin of NE
    #end_label -end of NE
    start_ne_pos = ipos
    ipos += 1
    while ipos < len(ipre_seq_label) and (ipre_seq_label[ipos] in end_label):
      ipos += 1
    end_ne_pos = ipos
    tmp_dict = {}
    tmp_dict['label_type'] = id2label[ipre_seq_label[start_ne_pos]]
    tmp_dict['overlap'] = 0
    tmp_dict['start_pos'] = token2orgindex[start_ne_pos] + 1
    tmp_dict['end_pos'] = token2orgindex[end_ne_pos - 1] + 1
    a_senten_res.append(tmp_dict)
    save_list.append(id2label[ipre_seq_label[start_ne_pos]] + '_' + org_text[token2orgindex[start_ne_pos] : token2orgindex[end_ne_pos - 1] + 1])
    return ipos

  def get_a_sentence_label_v2(self, save_list, ipre_seq_label, valid_len, org_text, token2orgindex, id2label, example=None):
    # 抽离出有效位置对应的文本
    a_senten_res = []
    ipre_seq_label = ipre_seq_label[1:valid_len-1]
    assert len(ipre_seq_label) == valid_len - 2

    ipos = 0
    while ipos < len(ipre_seq_label):
      if ipre_seq_label[ipos] not in [1,2,3,4]:
        ipos+=1
      else:
        # 起始位置
        if ipre_seq_label[ipos] == 1:
          ipos = self.get_sen_label_helper(1, [5,9], ipos, ipre_seq_label, id2label, token2orgindex, a_senten_res, save_list, org_text)
        elif ipre_seq_label[ipos] == 2:
          ipos = self.get_sen_label_helper(2, [6,10], ipos, ipre_seq_label, id2label, token2orgindex, a_senten_res, save_list, org_text)
        elif ipre_seq_label[ipos] == 3:
          ipos = self.get_sen_label_helper(3, [7,11], ipos, ipre_seq_label, id2label, token2orgindex, a_senten_res, save_list, org_text)
        elif ipre_seq_label[ipos] == 4:
          ipos = self.get_sen_label_helper(4, [8,12], ipos, ipre_seq_label, id2label, token2orgindex, a_senten_res, save_list, org_text)
    self.test_res.append(a_senten_res)
  
  def f1_score(self, pre_label, grou_true):
    pre_label=(set(pre_label))
    grou_true=(set(grou_true))
    print(list(pre_label)[:10])
    print(list(grou_true)[:10])
    AP=len(grou_true)
    TP=len(pre_label.intersection(grou_true))
    FP=len(pre_label)-TP
    print('ap len:', (AP))
    print('tp len:', (TP))
    print('fp len:', (FP))

    precision=TP/(TP+FP+1e-8)
    recall=TP/AP
    print('precision',precision)
    print('recall',recall)
    #print('差集：',len(grou_true-pre_label),grou_true-pre_label)
    #print('差集：',len(pre_label-grou_true),pre_label-grou_true)
    return 2*(precision*recall)/(1e-8+precision+recall)
  
  def get_test_res(self):
    return self.test_res, self.test_logit
  
  def score(self):
    return self.f1_score(self.final_type_strans_pairs, self.ground_truth_strans_pairs)

#   def get_a_sentence_label(self,save_list,ipre_seq_label,true_len,org_text,token2orgindex,id2label,example=None):
#     # 抽离出有效位置对应的文本

#     #保存字典（测试集需要）
#     a_senten_res=[]

#     tmp_len=len(save_list)

#     ipre_seq_label = ipre_seq_label[:true_len][1:-1]
#     previous_label_id = 0
#     point_start = -1
#     #print(ipre_seq_label)
#     #处理label的值，便于split
#     ipre_seq_label=self.change_ilabel(ipre_seq_label)
#     for i_pos, ipos_pre_label in enumerate(ipre_seq_label):

#       if ipos_pre_label != previous_label_id:
#         # 起始点
#         if previous_label_id == 0:
#           point_start = i_pos
#         else:
#           tmp_dict={}
#           tmp_dict['label_type']=id2label[previous_label_id]
#           tmp_dict['overlap']=0
#           tmp_dict['start_pos']=token2orgindex[point_start]+1
#           #包括end
#           tmp_dict['end_pos']=token2orgindex[i_pos]
#           a_senten_res.append(tmp_dict)
#           save_list.append(
#             id2label[previous_label_id] + '_' + org_text[token2orgindex[point_start]:token2orgindex[i_pos]])
#           point_start = i_pos

#       previous_label_id = ipos_pre_label

#       if i_pos == len(ipre_seq_label) - 1 and ipos_pre_label != 0:
#         tmp_dict = {}
#         tmp_dict['label_type'] = id2label[previous_label_id]
#         tmp_dict['overlap'] = 0
#         tmp_dict['start_pos'] = token2orgindex[point_start] + 1
#         # 包括end
#         tmp_dict['end_pos'] = token2orgindex[i_pos]
#         a_senten_res.append(tmp_dict)
#         save_list.append(id2label[ipos_pre_label] + '_' + org_text[token2orgindex[point_start]:])
#     if '“改进型海麻雀” Block 2' in org_text:
#       print('org_text:', org_text)
#       print(example.str_ans)
#       print(ipre_seq_label)
#       print(save_list[-len(save_list)+tmp_len:])
#       print(a_senten_res)

#     self.test_res.append(a_senten_res)



def set_seed(args):
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if args.n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)


class FGM():
  def __init__(self, model):
    self.model = model
    self.backup = {}

  def attack(self, epsilon=1., emb_name='bert.embeddings.word_embeddings.weight'):
    # emb_name这个参数要换成你模型中embedding的参数名
    for name, param in self.model.named_parameters():
      #             print(name, type(param),param)
      if param.requires_grad and emb_name in name:
        #                 print(name)
        self.backup[name] = param.data.clone()
        #                 print(param.grad)
        norm = torch.norm(param.grad)
        if norm != 0 and not torch.isnan(norm):
          r_at = epsilon * param.grad / norm
          param.data.add_(r_at)

  def restore(self, emb_name='bert.embeddings.word_embeddings.weight'):
    # emb_name这个参数要换成你模型中embedding的参数名
    for name, param in self.model.named_parameters():
      if param.requires_grad and emb_name in name:
        assert name in self.backup
        param.data = self.backup[name]
    self.backup = {}

class PGD_org():
  def __init__(self, model):
    self.model = model
    self.emb_backup = {}
    self.grad_backup = {}

  def attack(self, epsilon=1., alpha=0.3, emb_name='bert.embeddings.word_embeddings.weight', is_first_attack=False):
    # emb_name这个参数要换成你模型中embedding的参数名
    for name, param in self.model.named_parameters():
      if param.requires_grad and emb_name in name:
        if is_first_attack:
          self.emb_backup[name] = param.data.clone()
        norm = torch.norm(param.grad)
        if norm != 0 and not torch.isnan(norm):
          r_at = alpha * param.grad / norm
          param.data.add_(r_at)
          param.data = self.project(name, param.data, epsilon)

  def restore(self, emb_name='bert.embeddings.word_embeddings.weight'):
    # emb_name这个参数要换成你模型中embedding的参数名
    for name, param in self.model.named_parameters():
      if param.requires_grad and emb_name in name:
        assert name in self.emb_backup
        param.data = self.emb_backup[name]
    self.emb_backup = {}

  def project(self, param_name, param_data, epsilon):
    r = param_data - self.emb_backup[param_name]
    if torch.norm(r) > epsilon:
      r = epsilon * r / torch.norm(r)
    return self.emb_backup[param_name] + r

  def backup_grad(self):
    for name, param in self.model.named_parameters():
      if param.requires_grad:
        self.grad_backup[name] = param.grad.clone()

  def restore_grad(self):
    for name, param in self.model.named_parameters():
      if param.requires_grad:
        param.grad = self.grad_backup[name]
        
        
  # 问题集合
#   question_dict = \
#     {
#       '试验要素': '什么是实验要素？实验要素指的是试验鉴定工作的对象，如列为考核目标的武器装备（系统级）、技术、战术、人员、对象之间的能力等；支持完成试验鉴定所需的条件，如陪试品、参试装备、测试、测量、靶标、仿真等；装备的基本情况等。\
#   例如：RS-24弹道导弹、SPY-1D相控阵雷达、紫菀防空导弹（Aster）、F-35“闪电”II型联合攻击战斗机、“阿利·伯克”级Flight IIA型驱逐舰“约翰芬”号、协同通信与指挥、连续波测量雷达、电影经纬仪、无人机靶标等',
#       '性能指标': '什么是性能指标？性能指标指的是试验要素在技术、使用等性能方面的定性、定量描述，如重量、射程、可靠性等。\
#   例如：测量精度、圆概率偏差、失效距离、准备时间、反激光毁伤、发射方式等。',
#       '系统组成': '什么是系统组成？系统组成指的是被试对象的组成部分，如子系统、部件、采用的技术等。\
#   例如：动能杀伤飞行器（KKV）、中波红外导引头、助推器、整流罩、箔条红外混合诱饵弹、碰撞杀伤技术、柔性摆动喷管技术、端羟基聚丁二烯、等。',
#       '任务场景': '什么是任务场景？任务场景指的是试验要素在发挥其实际效用和价值中涉及的信息，如人员、对抗目标、体系能力等。\
#   例如：法国海军、导弹预警、恐怖袭击、迫击炮威胁、排级作战等。'
#     }

  # 不同类型数据分割
#   example_start = example.start
#   example_end = example.end
#   example_label = example.label_type
#   example_strans = example.str_ans
#   if is_training in [1, 2]:
#     for istart,iend,ilabel,istrans in zip(example_start,example_end,example_label,example_strans):
#       #不在文本中的NE去掉
# #       max_doc_length = max_seq_length - len(tokenizer.tokenize(question_dict[ilabel])[:max_question_length]) - 3
#       if iend>=max_doc_length:
#         continue
#       if ilabel in selected_sentenses.keys():
#         selected_sentenses[ilabel].append({
#           'tokens': passage_tokens[:max_doc_length],
#           'start': 0, 'end': len(passage_tokens) - 1,'ilabel':ilabel,'istrans':istrans
#         })
#       else:
#         selected_sentenses[ilabel]=[]
#         selected_sentenses[ilabel].append({
#           'tokens': passage_tokens[:max_doc_length],
#           'start': 0, 'end': len(passage_tokens) - 1, 'ilabel': ilabel, 'istrans': istrans
#         })
#   #每个类型的句子都有
#   if len(selected_sentenses)!=len(question_dict):
#     for itypekey in question_dict.keys():
#       if itypekey not in selected_sentenses.keys():
#         #doc截取
#         max_doc_length = max_seq_length - len(tokenizer.tokenize(question_dict[itypekey])[:max_question_length]) - 3
#         selected_sentenses[itypekey] = []
#         selected_sentenses[itypekey].append({
#           'tokens': passage_tokens[:max_doc_length],
#           'start': -1, 'end': -1, 'ilabel': -1, 'istrans': -1
#         })
#   assert len(selected_sentenses) > 0, f'len(selected_sentenses) is 0!'

#   feature_count = 0

#   #while i < len(selected_sentenses):
#   for itypekey,iselected_sentenses in selected_sentenses.items():
#     print('selected_sentenses.keys()',selected_sentenses.keys())
#     label = [0] * max_seq_length
#     #最大长度设置
#     question_tokens=tokenizer.tokenize(question_dict[itypekey])[:max_question_length]
#     quention_len=len(question_tokens)
#     max_doc_length = max_seq_length - quention_len - 3
#     for i in range(len(iselected_sentenses)):
#       # 当前句子长度超多最大允许长度，按照滑动窗口切分,(当前只有一个句子)
#       for sentense_start in range(0, len(iselected_sentenses[i]['tokens']), 512):
#         sentense_end = min(sentense_start + max_doc_length - 1, len(iselected_sentenses[i]['tokens']) - 1)
#         multi_sentense_start = iselected_sentenses[i]['start'] + sentense_start
#         multi_sentense_end = iselected_sentenses[i]['start'] + sentense_end

#         if is_training in [1, 2]:
#           if ilabel ==-1:
#             continue
#           ilabel,istart,iend,istr_ans = \
#             iselected_sentenses[i]['ilabel'],\
#           iselected_sentenses[i]['start'],\
#           iselected_sentenses[i]['end'],\
#           iselected_sentenses[i]['istrans']

#           start_position = original_to_tokenized_index[istart]
#           end_position = original_to_tokenized_index[iend]

#           start = start_position - multi_sentense_start + 2+quention_len
#           end = end_position - multi_sentense_start + 2+quention_len

#           #B I E label

#           label[start:end+1]=[label_map[ilabel]]+[label_map[ilabel]+4]*(end-start)
#           #if (end+1)-start>1:
#           #  label[end]=label_map[ilabel]+8

#           #assert istr_ans==''.join(selected_sentenses[i]['tokens'][start_position:end_position+1]),\
#           #  (istr_ans,''.join(selected_sentenses[i]['tokens'][start_position:end_position+1]))
#           assert -1 <= end < max_seq_length - 1, f'end position is out of range: {end}'
#     feature_count += 1
#     doc_tokens = selected_sentenses[i]['tokens'][sentense_start:sentense_end + 1]
#     assert len(doc_tokens) <= max_doc_length, f'len(doc_tokens) too much:{len(doc_tokens)}'
#     input_tokens = ['[CLS]'] +question_tokens+['[SEP]']+ doc_tokens + ['[SEP]']
#     bert_doc_end = len(input_tokens) - 1
#     assert len(input_tokens) <= max_seq_length
#     segment_ids = [0]*(1+len(question_tokens)+1)+[1] * (len(doc_tokens) + 1)
#     input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
#     assert  len(input_tokens)==len(input_ids)
#     input_mask = [1] * len(input_ids)
#     # padding
#     padding_length = max_seq_length - len(input_ids)
#     input_ids += ([0] * padding_length)
#     input_mask += ([0] * padding_length)
#     segment_ids += ([0] * padding_length)
#     # print(label)
#     if is_training == 1:  # train dataset
#       features.append(
#         Feature(
#           example_id=example.guid,
#           doc_id=example.docid,
#           doc_start=multi_sentense_start,
#           doc_end=multi_sentense_end,
#           doc_length=multi_sentense_end - multi_sentense_start,
#           question_len=len(question_tokens),
#           tokenized_to_original_index=tokenized_to_original_index,
#           input_tokens=input_tokens,
#           input_ids=input_ids,
#           input_mask=input_mask,
#           segment_ids=segment_ids,
#           bert_start_position=None,
#           bert_end_position=None,
#           bert_answer_span=None,
#           label=example.label_type,
#           token_start=start_position,
#           token_end=end_position,
#           str_ans=example.str_ans,
#           label_id=label
#         ))
#     elif is_training == 2:  # eval dataset
#       features.append(
#         Feature(
#           example_id=example.guid,
#           doc_id=example.docid,
#           passage=example.text_a,
#           doc_start=multi_sentense_start,
#           doc_end=multi_sentense_end,
#           bert_doc_end=bert_doc_end,
#           question_len=len(question_tokens),
#           question=example.text_b,
#           tokenized_to_original_index=tokenized_to_original_index,
#           input_tokens=input_tokens,
#           input_ids=input_ids,
#           input_mask=input_mask,
#           segment_ids=segment_ids,
#           bert_start_position=None,
#           bert_end_position=None,
#           bert_answer_span=None,
#           label=example.label_type,
#           origin_start=example.start,
#           origin_end=example.end,
#           token_start=start_position,
#           token_end=end_position,
#           str_ans=example.str_ans,
#           label_id=label
#         ))
#     else:  # test
#       features.append(
#         Feature(
#           example_id=example.guid,
#           doc_id=example.docid,
#           passage=example.text_a,
#           doc_start=multi_sentense_start,
#           doc_end=multi_sentense_end,
#           bert_doc_end=bert_doc_end,
#           question_len=len(question_tokens),
#           question=example.text_b,
#           tokenized_to_original_index=tokenized_to_original_index,
#           input_tokens=input_tokens,
#           input_ids=input_ids,
#           input_mask=input_mask,
#           segment_ids=segment_ids,
#           bert_start_position=None,
#           bert_end_position=None,
#           bert_answer_span=None,
#           label=example.label_type,
#           origin_start=example.start,
#           origin_end=example.end,
#           token_start=None,
#           token_end=None,
#           str_ans=example.str_ans,
#           label_id=label
#         ))