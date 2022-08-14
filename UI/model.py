import os
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")

sys.path.append('/home/primedo/kaggle/tweet_sentiment_extraction/tweet-extraction')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from tqdm import tqdm_notebook as tqdm
from torch import nn
import copy
import transformers
print(transformers.__version__)
from transformers.models.t5.modeling_t5 import *
import json
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
# from transformers import *
from transformers import BartTokenizer, RobertaTokenizer,  AdamW, get_linear_schedule_with_warmup, get_constant_schedule
import torch
# import matplotlib.pyplot as plt
import torch.utils.data
import torch.nn.functional as F
from torch.optim import Adagrad, Adamax
from transformers.modeling_utils import *
from scipy.special import softmax
import argparse
import itertools
from collections import OrderedDict
from fuzzywuzzy import fuzz
import operator
## Import from tweet-extraction
from models import *
from transformers.models.bert.tokenization_bert import BasicTokenizer
# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm

batch_size = 1
beam_size = 3
max_sequence_length = 128

def find_best_combinations(start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, valid_start= 0, valid_end=512):
    best = (valid_start, valid_end - 1)
    best_score = -9999
    for i in range(len(start_top_log_probs)):
        for j in range(end_top_log_probs.shape[0]):
            if valid_start <= start_top_index[i] < valid_end and valid_start <= end_top_index[j,i] < valid_end and start_top_index[i] < end_top_index[j,i]:
                score = start_top_log_probs[i] * end_top_log_probs[j,i]
                if score > best_score:
                    best = (start_top_index[i],end_top_index[j,i])
                    best_score = score
    return best


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    try:
        return float(len(c)) / (len(a) + len(b) - len(c))
    except:
        return 0

def fuzzy_match(x,y,weights=None):
    l1 = len(x.split())
    matches = dict()
    x_ = x.split()
    if type(y) is str:
        y = [y]
    for curr_length in range(l1 + 1):
        for i in range(l1 + 1 - curr_length):
            sub_x = ' '.join(x_[i:i+curr_length])
            if sub_x not in matches:
                matches[sub_x] = np.average([fuzz.ratio(sub_x,y_) for y_ in y],weights=weights)
    if len(matches) == 0:
        return None, x
    return matches, sorted(matches.items(), key=operator.itemgetter(1))[-1][0]
    
def ensemble_v0(context, predictions):
    context = context.split()
    scores = dict()
    for i,j in itertools.combinations(range(len(context) + 1),r=2):
        curr_context = ' '.join(context[i:j])
        scores[curr_context] = np.mean([jaccard(curr_context, p) for p in predictions])
    best_score = np.max([val for val in scores.values()])
    has_tie = np.sum([val == best_score for val in scores.values()]) > 1
    if not has_tie:
        for key, val in scores.items():
            if val == best_score:
                return key
    else:
        keys = [key for key, val in scores.items() if val == best_score]
#         return keys[np.argmax([jaccard(key,predictions[0]) for key in keys])]
        return keys[np.argmax([len(key.split()) for key in keys])]

def ensemble(context, predictions):
    starts = []
    ends = []
    for p in predictions:
        if p in context:
            start = context.index(p)
            starts.append(start)
            ends.append(start+len(p))
    if len(starts) == 0:
        print(context)
        return ensemble_v0(context, predictions)
    scores = dict()
    context = context[np.min(starts):np.max(ends)]
    for i,j in itertools.combinations(range(len(context) + 1),r=2):
        if len(context.split()) == 1 or \
            (not ((i > 0 and context[i-1].isalnum() and context[i].isalnum()) or (j < len(context) and j > 1 and context[j-1].isalnum() and context[j].isalnum()))):
            curr_context = context[i:j].strip()
            scores[curr_context] = np.mean([jaccard(curr_context, p) for p in predictions])
    for pred in predictions:
        if pred not in scores:
            scores[pred] = np.mean([jaccard(pred, p) for p in predictions])
    best_score = np.max([val for val in scores.values()])
    has_tie = np.sum([val == best_score for val in scores.values()]) > 1
    if not has_tie:
        for key, val in scores.items():
            if val == best_score:
                return key
    else:
        keys = [key for key, val in scores.items() if val == best_score]
#         return keys[np.argmax([jaccard(key,predictions[0]) for key in keys])]
        return keys[np.argmax([len(key.split()) for key in keys])]

basic_tokenizer = BasicTokenizer(do_lower_case=False)
def fix_spaces(t):
    for i,item in enumerate(t):
        re_res = re.search('\s+$', item)
        if bool(re_res) & (i < len(t)-1):
            sp = re_res.span()
            t[i+1] = t[i][sp[0]:] + t[i+1]
            t[i] = t[i][:sp[0]]
    return t
def roberta_tokenize_v2(tokenizer, line):
    tokenized_line = []
    line2 = basic_tokenizer._run_split_on_punc(line)
    line2 = fix_spaces(line2)
    for item in line2:
        sub_word_tokens = tokenizer.tokenize(item)
        tokenized_line += sub_word_tokens
    return tokenized_line


def convert_lines(tokenizer, df, max_sequence_length = 512):
    pad_token_idx = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    outputs = np.zeros((len(df), max_sequence_length))
    type_outputs = np.zeros((len(df), max_sequence_length))
    position_outputs = np.zeros((len(df), 2))
    extracted = []
    for idx, row in tqdm(df.iterrows(), total=len(df)): 
        input_ids_0 = tokenizer.convert_tokens_to_ids(roberta_tokenize_v2(tokenizer, row.sentiment)) 
        input_ids_1 = tokenizer.convert_tokens_to_ids(roberta_tokenize_v2(tokenizer, row.text)) 
        input_ids = [tokenizer.cls_token_id, ]+ input_ids_0 +  [tokenizer.sep_token_id,] +input_ids_1 + [tokenizer.sep_token_id, ]
        token_type_ids = [0,]*(len(input_ids_0) + 1) + [1,]*(len(input_ids_1) + 2)
        if len(input_ids) > max_sequence_length: 
#            input_ids = input_ids[:max_sequence_length//2] + input_ids[-max_sequence_length//2:] 
            input_ids = input_ids[:max_sequence_length]
            input_ids[-1] = tokenizer.sep_token_id
            token_type_ids = token_type_ids[:max_sequence_length]
        else:
            input_ids = input_ids + [pad_token_idx, ]*(max_sequence_length - len(input_ids))
            token_type_ids = token_type_ids + [pad_token_idx, ]*(max_sequence_length - len(token_type_ids))
        assert len(input_ids) == len(token_type_ids)
        outputs[idx,:max_sequence_length] = np.array(input_ids)
        type_outputs[idx,:] = token_type_ids
    return outputs, type_outputs


def convert_data(tokenizer, text, sentiment, max_sequence_length = 512):
    pad_token_idx = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    outputs = np.zeros((1, max_sequence_length))
    type_outputs = np.zeros((1, max_sequence_length))
    position_outputs = np.zeros((1, 2))
    extracted = []
    
    input_ids_0 = tokenizer.convert_tokens_to_ids(roberta_tokenize_v2(tokenizer, sentiment)) 
    input_ids_1 = tokenizer.convert_tokens_to_ids(roberta_tokenize_v2(tokenizer, text)) 
    input_ids = [tokenizer.cls_token_id, ]+ input_ids_0 +  [tokenizer.sep_token_id,] +input_ids_1 + [tokenizer.sep_token_id, ]
    token_type_ids = [0,]*(len(input_ids_0) + 1) + [1,]*(len(input_ids_1) + 2)
    if len(input_ids) > max_sequence_length: 
#            input_ids = input_ids[:max_sequence_length//2] + input_ids[-max_sequence_length//2:] 
        input_ids = input_ids[:max_sequence_length]
        input_ids[-1] = tokenizer.sep_token_id
        token_type_ids = token_type_ids[:max_sequence_length]
    else:
        input_ids = input_ids + [pad_token_idx, ]*(max_sequence_length - len(input_ids))
        token_type_ids = token_type_ids + [pad_token_idx, ]*(max_sequence_length - len(token_type_ids))
    assert len(input_ids) == len(token_type_ids)
    outputs[0,:max_sequence_length] = np.array(input_ids)
    type_outputs[0,:] = token_type_ids

    return outputs, type_outputs

def get_predictions(x_test, x_type_test, model,tokenizer, is_xlnet=False):
    all_start_top_log_probs = None
    all_start_top_index = None
    all_end_top_log_probs = None
    all_end_top_index = None
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(x_test,dtype=torch.long), torch.tensor(x_type_test,dtype=torch.long))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        pbar = tqdm(enumerate(test_loader),total=len(test_loader),leave=False)
        for i, items in pbar:
            x_batch, x_type_batch = items
            attention_mask = x_batch != tokenizer.pad_token_id
            p_mask = torch.zeros(x_batch.shape,dtype=torch.float32)
            p_mask[x_batch == tokenizer.pad_token_id] = 1.0
            p_mask[x_batch == tokenizer.cls_token_id] = 1.0
            if is_xlnet:
                attention_mask = attention_mask.float()
                p_mask[:,:2] = 1.0
            else:
                p_mask[:,:3] = 1.0
            start_top_log_probs, start_top_index, end_top_log_probs, end_top_index = model(input_ids=x_batch.cuda(), attention_mask=attention_mask.cuda(), \
                                                token_type_ids=x_type_batch.cuda(), beam_size=beam_size, p_mask=p_mask.cuda())
            start_top_log_probs = start_top_log_probs.detach().cpu().numpy()
            start_top_index = start_top_index.detach().cpu().numpy()
            end_top_log_probs = end_top_log_probs.detach().cpu().numpy()
            end_top_index = end_top_index.detach().cpu().numpy()

            all_start_top_log_probs = start_top_log_probs if all_start_top_log_probs is None else np.concatenate([all_start_top_log_probs, start_top_log_probs])
            all_start_top_index = start_top_index if all_start_top_index is None else np.concatenate([all_start_top_index, start_top_index])
            all_end_top_log_probs = end_top_log_probs if all_end_top_log_probs is None else np.concatenate([all_end_top_log_probs, end_top_log_probs])
            all_end_top_index = end_top_index if all_end_top_index is None else np.concatenate([all_end_top_index, end_top_index])

    return all_start_top_log_probs,all_start_top_index,all_end_top_log_probs,all_end_top_index

def load_and_fix_state(model_path):
    state_dict = torch.load(model_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

def get_selected_texts(raw_x, x, tokenizer, all_start_top_log_probs, all_start_top_index, all_end_top_log_probs, all_end_top_index, offset=3, is_xlnet=False):
    selected_texts = []
    
    real_length = np.sum(x != tokenizer.pad_token_id)
    if is_xlnet:
        real_length -= 1
    best_start, best_end = find_best_combinations(all_start_top_log_probs[0], all_start_top_index[0], \
                                                    all_end_top_log_probs[0].reshape(beam_size,beam_size), all_end_top_index[0].reshape(beam_size,beam_size), \
                                                    valid_start = offset, valid_end = real_length)
#         selected_text = tokenizer.decode([w for w in x[best_start:best_end] if w != tokenizer.pad_token_id]).strip()
#         selected_text = " ".join(selected_text.split()).lower()
#         selected_texts.append(selected_text.lower() if selected_text in test_df.loc[i_].sep_text 
#                               else fuzzy_match(test_df.loc[i_].sep_text, selected_text)[-1])
    selected_text = tokenizer.decode([w for w in x[best_start:best_end] if w != tokenizer.pad_token_id],clean_up_tokenization_spaces=False)
    selected_texts.append(selected_text if selected_text in raw_x 
                            else fuzzy_match(raw_x, selected_text)[-1])
    return selected_texts

def check_corrs(model_name):
    corrs = np.zeros((5,5))
    for i in range(5):
        for j in range(5):
            if corrs[j,i] == 0:
                corrs[i,j] = np.mean([jaccard(x,y) for x,y in zip(all_preds[f"{model_name}_{i}"], all_preds[f"{model_name}_{j}"])])
                corrs[j,i] = corrs[i,j]
    return corrs

model_dir = "/media/aiteam/storage/AI_DATA/data/kaggle_data/tweet_sentiment_extraction"
all_preds = dict()

def init_roberta(name):
    tokenizer = RobertaTokenizer.from_pretrained(name)
    model = RobertaForSentimentExtraction.from_pretrained(name, output_hidden_states=True)
    model.cuda()
    return model, tokenizer

def init_bart(name):
    tokenizer = BartTokenizer.from_pretrained(name)
    model = BartForSentimentExtraction.from_pretrained(name, output_hidden_states=True)
    model.cuda()
    return model, tokenizer


def initialize():
    models = {}
    tokenizers = {}
    names = ['roberta-base', 'roberta-large', 'facebook/bart-large']
    for name in names:
        if 'bart' not in name:
            models[name], tokenizers[name] = init_roberta(name)
        else:
            models[name], tokenizers[name] = init_bart(name)
    return models, tokenizers
    


def predict(raw_text, sentiment, models, tokenizers):
    text = " ".join(raw_text.split()).lower()
    tokenizer = tokenizers['roberta-base']
    model = models['roberta-base']
    # X_test, X_type_test = convert_lines(tokenizer, test_df, max_sequence_length= max_sequence_length)
    x, type_x = convert_data(tokenizer, raw_text, sentiment, max_sequence_length= max_sequence_length)
    # seeds = [13,23,33]
    seeds = [14,24,34,44,54]

    for seed, fold in zip(seeds, np.arange(5)):
        model.load_state_dict(torch.load(f"{model_dir}/models/roberta_{fold}_{seed}.bin"), strict=False)
        model.eval()

        # all_start_top_log_probs,all_start_top_index,all_end_top_log_probs,all_end_top_index
        all_outs = get_predictions(x, type_x, model, tokenizer)
        selected_texts = get_selected_texts(raw_text, x.squeeze(), tokenizer, *all_outs)
        all_preds[f"roberta_{fold}_{seed}"] = [x if type(x) is str else "" for x in selected_texts]

    tokenizer = tokenizers['roberta-large']
    model = models['roberta-large']
    x, type_x = convert_data(tokenizer, raw_text, sentiment, max_sequence_length= max_sequence_length)
    # seeds = [43,53,63]
    seeds = [64,74,84,94,104]
    # for seed in seeds:
    #     for fold in range(5):
    for seed, fold in zip(seeds, np.arange(5)):
        model.load_state_dict(torch.load(f"{model_dir}/models/roberta-large_{fold}_{seed}.bin"), strict=False)
        model.eval()
        all_outs = get_predictions(x, type_x, model, tokenizer)
        selected_texts = get_selected_texts(raw_text, x.squeeze(), tokenizer, *all_outs)
        all_preds[f"roberta-large_{fold}_{seed}"] = [x if type(x) is str else "" for x in selected_texts]


    tokenizer = tokenizers['facebook/bart-large']
    model = models['facebook/bart-large']
    x, type_x = convert_data(tokenizer, raw_text, sentiment, max_sequence_length= max_sequence_length)
    # seeds = [73,83,93]
    seeds = [114,124,134,144,154]
    # for seed in seeds:
    #     for fold in range(5):
    for seed, fold in zip(seeds, np.arange(5)):
        model.load_state_dict(torch.load(f"{model_dir}/models/bart-large_{fold}_{seed}.bin"), strict=False)
        model.eval()
        all_outs = get_predictions(x, type_x, model, tokenizer)
        selected_texts = get_selected_texts(raw_text, x.squeeze(), tokenizer, *all_outs)
        all_preds[f"bart-large_{fold}_{seed}"] = [x if type(x) is str else "" for x in selected_texts]

    model_list = [key for key in all_preds.keys()]

    all_vals = [all_preds[model_name] for model_name in model_list]
    ensembled = []
    # sep_texts = test_df.text.apply(lambda x: " ".join(x.split()).lower())

    predictions = [val[0] for val in all_vals]
    return ensemble(raw_text, predictions)
    


if __name__ == "__main__":
    text = 'Shanghai is also really exciting (precisely -- skyscrapers galore). Good tweeps in China:  (SH)  (BJ).'
    predict(text, "positive")