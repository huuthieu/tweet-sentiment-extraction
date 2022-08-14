from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary
from types import SimpleNamespace
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm 
import regex as re
import html
import operator
from fuzzywuzzy import fuzz
import itertools
from scipy.special import softmax
# from transformers.tokenization_bert import BasicTokenizer
from transformers.models.bert.tokenization_bert import BasicTokenizer
import math 

def jaccard(str1, str2): 
    if type(str1) is str:
        a = set(str1.lower().split()) 
        b = set(str2.lower().split())
    else:
        a = set(str1)
        b = set(str2)
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
#        if curr_length <= 0:
#            continue
        for i in range(l1 + 1 - curr_length):
            sub_x = ' '.join(x_[i:i+curr_length])
            if sub_x not in matches:
                matches[sub_x] = np.average([fuzz.ratio(sub_x,y_) for y_ in y],weights=weights)
#                matches[sub_x] = jaccard(sub_x,y)
    if len(matches) == 0:
        return None, x
    return matches, sorted(matches.items(), key=operator.itemgetter(1))[-1][0]

def contains(small, big):
    for i in range(len(big)-len(small)+1):
        for j in range(len(small)):
            if big[i+j] != small[j]:
                break
        else:
            return i, i+len(small)
    return None

def get_sub_idx(x, y): 
#    best_span = contains(y,x)
#    if best_span is not None:
#        return best_span
    l1, l2 = len(x), len(y) 
    best_span = None
    best_score = 0.5
    for i in range(l1):
        for j in range(i+1, l1+1):
            score = fuzz.ratio(x[i:j],y)
            if score > best_score:
                best_span = (i,j)
                best_score = score
    if best_span is None:
        return 0,l1
    else:
        return best_span

def find_best_combinations(start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, valid_start= 0, valid_end=512):
    best = (valid_start, valid_end - 1)
    best_score = -9999
#    print(valid_end, start_top_index, end_top_index)
    for i in range(len(start_top_log_probs)):
        for j in range(end_top_log_probs.shape[0]):
            if valid_start <= start_top_index[i] < valid_end and valid_start <= end_top_index[j,i] < valid_end and start_top_index[i] < end_top_index[j,i]:
                score = start_top_log_probs[i] * end_top_log_probs[j,i]
                if score > best_score:
                    best = (start_top_index[i],end_top_index[j,i])
                    best_score = score
    return best

special_tokens = {"positive": "[POS]", "negative":"[NEG]", "neutral": "[NTR]"}

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

class BERTweetTokenizer():
    
    def __init__(self,pretrained_path = './bertweet/'):
        

        self.bpe = fastBPE(SimpleNamespace(bpe_codes= pretrained_path + "bpe.codes"))
        self.vocab = Dictionary()
        self.vocab.add_from_file(pretrained_path + "dict.txt")
        self.cls_token_id = 0
        self.pad_token_id = 1
        self.sep_token_id = 2
        self.pad_token = '<pad>'
        self.cls_token = '<s>'
        self.sep_token = '</s>'
        
    def bpe_encode(self,text):
        return self.bpe.encode(text)
    
    def encode(self,text,add_special_tokens=False):
        subwords = self.bpe.encode(text)
        input_ids = self.vocab.encode_line(subwords, append_eos=False, add_if_not_exist=False).long().tolist()
        return input_ids
    
    def tokenize(self,text):
        return self.bpe_encode(text).split()
    
    def convert_tokens_to_ids(self,tokens):
        input_ids = self.vocab.encode_line(' '.join(tokens), append_eos=False, add_if_not_exist=False).long().tolist()
        return input_ids
    
    def decode(self, ids, clean_up_tokenization_spaces=False):
        return self.vocab.string(ids, bpe_symbol = '@@')

def convert_lines_v2(tokenizer, df, max_sequence_length = 512):
    pad_token_idx = tokenizer.pad_token_id
    outputs = np.zeros((len(df), max_sequence_length))
    type_outputs = np.zeros((len(df), max_sequence_length))
    position_outputs = np.zeros((len(df), 2))
    offset_outputs = np.zeros((len(df),))
    extracted = []
    for idx, row in tqdm(df.iterrows(), total=len(df)): 
        input_ids_0 = tokenizer.convert_tokens_to_ids(roberta_tokenize_v2(tokenizer, row.sentiment)) 
        input_ids_1 = tokenizer.convert_tokens_to_ids(roberta_tokenize_v2(tokenizer, row.text)) 
        input_ids = [tokenizer.cls_token_id, ]+ input_ids_0 +  [tokenizer.sep_token_id,] +input_ids_1 + [tokenizer.sep_token_id, ]
        token_type_ids = [0,]*(len(input_ids_0) + 1) + [1,]*(len(input_ids_1) + 2)

        if len(input_ids) > max_sequence_length: 
            input_ids = input_ids[:max_sequence_length]
            input_ids[-1] = tokenizer.sep_token_id
            token_type_ids = token_type_ids[:max_sequence_length]
        else:
            input_ids = input_ids + [pad_token_idx, ]*(max_sequence_length - len(input_ids))
            token_type_ids = token_type_ids + [pad_token_idx, ]*(max_sequence_length - len(token_type_ids))
        assert len(input_ids) == len(token_type_ids)
        outputs[idx,:max_sequence_length] = np.array(input_ids)
        type_outputs[idx,:] = token_type_ids
        selected_text = row.selected_text.strip()
        if " "+selected_text in row.text:
            input_ids_2 = tokenizer.convert_tokens_to_ids(roberta_tokenize_v2(tokenizer," "+selected_text))
        else:
            input_ids_2 = tokenizer.convert_tokens_to_ids(roberta_tokenize_v2(tokenizer,selected_text))
        start_idx, end_idx = get_sub_idx(input_ids_1, input_ids_2)
#        extracted.append(tokenizer.decode(input_ids_1[start_idx:end_idx]))
        position_outputs[idx,:] = [start_idx + len(input_ids_0) + 2, end_idx + len(input_ids_0) + 2]
        offset_outputs[idx] = len(input_ids_0) + 2
#    df["extracted"] = extracted
#    df[["text","selected_text","extracted"]].to_csv("extracted.csv",index=False)
    return outputs, type_outputs, position_outputs, offset_outputs

def convert_lines_xlnet(tokenizer, df, max_sequence_length = 512):
    pad_token_idx = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    outputs = np.zeros((len(df), max_sequence_length))
    type_outputs = np.zeros((len(df), max_sequence_length))
    position_outputs = np.zeros((len(df), 2))
    extracted = []
    for idx, row in tqdm(df.iterrows(), total=len(df)): 
        input_ids_0 = tokenizer.convert_tokens_to_ids(roberta_tokenize_v2(tokenizer, row.sentiment)) 
        input_ids_1 = tokenizer.convert_tokens_to_ids(roberta_tokenize_v2(tokenizer, row.text)) 
        input_ids =  input_ids_0 +  [tokenizer.sep_token_id, ] + input_ids_1 + [tokenizer.sep_token_id, ] + [tokenizer.cls_token_id, ]
        token_type_ids = [0,]*(len(input_ids_0) + 1) + [1,]*(len(input_ids_1) + 2)

        if len(input_ids) > max_sequence_length: 
#            input_ids = input_ids[:max_sequence_length//2] + input_ids[-max_sequence_length//2:] 
            input_ids = input_ids[:max_sequence_length]
            input_ids[-2] = tokenizer.sep_token_id
            input_ids[-1] = tokenizer.cls_token_id
            token_type_ids = token_type_ids[:max_sequence_length]
        else:
            input_ids = input_ids + [pad_token_idx, ]*(max_sequence_length - len(input_ids))
            token_type_ids = token_type_ids + [pad_token_idx, ]*(max_sequence_length - len(token_type_ids))
        assert len(input_ids) == len(token_type_ids)
        outputs[idx,:max_sequence_length] = np.array(input_ids)
        type_outputs[idx,:] = token_type_ids
        selected_text = row.selected_text.strip()
        if " "+selected_text in row.text:
            input_ids_2 = tokenizer.convert_tokens_to_ids(roberta_tokenize_v2(tokenizer," "+selected_text))
        else:
            input_ids_2 = tokenizer.convert_tokens_to_ids(roberta_tokenize_v2(tokenizer,selected_text))
        start_idx, end_idx = get_sub_idx(input_ids_1, input_ids_2)
        extracted.append(tokenizer.decode(input_ids_1[start_idx:end_idx]))
        position_outputs[idx,:] = [start_idx + len(input_ids_0) + 1, end_idx + len(input_ids_0) + 1]
    type_outputs[outputs == tokenizer.cls_token_id] = 2
    type_outputs[outputs == pad_token_idx] = 4
    return outputs, type_outputs, position_outputs

def convert_lines_cls(tokenizer, df, max_sequence_length = 512):
    pad_token_idx = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    outputs = np.zeros((len(df), max_sequence_length))
    type_outputs = np.zeros((len(df), max_sequence_length))
    for idx, row in tqdm(df.iterrows(), total=len(df)): 
        input_ids = tokenizer.encode(row.text, add_special_tokens=False) 
        input_ids = [tokenizer.cls_token_id, ]+ input_ids +  [tokenizer.sep_token_id, ] 
        token_type_ids = [0,]*(len(input_ids)) 

        if len(input_ids) > max_sequence_length: 
#            input_ids = input_ids[:max_sequence_length//2] + input_ids[-max_sequence_length//2:] 
            input_ids = input_ids[:max_sequence_length]
            input_ids[-1] = tokenizer.sep_token_id
            token_type_ids = token_type_ids[:max_sequence_length]
        else:
            input_ids = input_ids + [pad_token_idx, ]*(max_sequence_length - len(input_ids))
            token_type_ids = token_type_ids + [pad_token_idx, ]*(max_sequence_length - len(token_type_ids))
        if len(input_ids) != len(token_type_ids):
            print(input_ids, token_type_ids)
        outputs[idx,:max_sequence_length] = np.array(input_ids)
        type_outputs[idx,:] = token_type_ids
    return outputs, type_outputs
