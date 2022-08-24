import os
import random
import numpy as np
import pandas as pd
import logging

from nltk.translate.bleu_score import sentence_bleu
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from rouge import rouge_w_sentence_level
from tqdm import tqdm

logger = logging.getLogger(__name__)

class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target


def read_examples(filename):
    """Read examples from filename."""
    examples = []
    df = pd.read_csv(filename)
    code = df['func_name'].tolist()
    nl = df['nl'].tolist()
    for i in range(len(code)):
        examples.append(
            Example(
                idx=i,
                source=nl[i].lower(),
                target=code[i].lower(),
            )
        )
    return examples


def read_examples_train(filename):
    """Read examples from filename."""
    examples = []
    df = pd.read_csv(filename)
    code = df['func_name'].tolist()
    nl = df['nl'].tolist()
    for i in range(len(code)):
        examples.append(
            Example(
                idx=i,
                source=nl[i].lower(),
                target=code[i].lower(),
            )
        )
    return examples

class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids


def convert_examples_to_features(examples, tokenizer, max_source_length, max_target_length, stage=None):
    features = []
    for example_index, example in enumerate(tqdm(examples, desc='convert examples to features...')):
        # source
        source_tokens = tokenizer.tokenize(example.source)[:max_source_length - 5]
        source_tokens = [tokenizer.cls_token, "<encoder-decoder>", tokenizer.sep_token] + source_tokens + ["<mask0>",tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length

        # target
        if stage == "test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:max_target_length - 2]
        target_tokens = ["<mask0>"] + target_tokens + [tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        padding_length = max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length

        if example_index < 3:
            if stage == 'train':
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example.idx))

                logger.info("source_tokens: {}".format([x.replace('\u0120', '_') for x in source_tokens]))
                logger.info("source_ids: {}".format(' '.join(map(str, source_ids))))

                logger.info("target_tokens: {}".format([x.replace('\u0120', '_') for x in target_tokens]))
                logger.info("target_ids: {}".format(' '.join(map(str, target_ids))))

        features.append(
            InputFeatures(
                example_index,
                source_ids,
                target_ids
            )
        )
    return features


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def rouge_w(prediction: str, ground_truth: str):
    gt_tokens = ground_truth.strip().split(" ")
    pre_tokens = prediction.strip().split(" ")
    # ROUGE-W
    recall, precision, f1 = rouge_w_sentence_level(pre_tokens, gt_tokens)
    return recall


def predict(source, max_source_length, model, tokenizer, beam_size):
    encode = tokenizer.encode_plus(source, return_tensors="pt", max_length=max_source_length, truncation=True,
                                        pad_to_max_length=True)
    source_ids = encode['input_ids'].cuda()

    model.eval()
    result_list = []
    with torch.no_grad():
        summary_text_ids = model(source_ids=source_ids)
        t = summary_text_ids[0][0].cpu().numpy()
        for i in range(beam_size):
            t = summary_text_ids[0][i].cpu().numpy()
            text = tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            result_list.append(text)
    return list(set(result_list))[:3]

def find_mixed_nn(simi, diffs, test_diff, bleu_thre, alpha):
    candidates = simi.argsort()[-bleu_thre:][::-1]
    max_score = 0
    max_idx = 0
    for j in candidates:
        bleu_score = sentence_bleu([diffs[j].split()], test_diff.split())
        rouge_score = rouge_w(diffs[j], test_diff)
        score = alpha*bleu_score + (1-alpha)*rouge_score
        if score > max_score:
            max_score = score
            max_idx = j
    return max_idx

def nngen(train_codes, train_nls, test_codes, bleu_thre, alpha):
    counter = TfidfVectorizer()
    train_matrix = counter.fit_transform(train_codes)
    test_matrix = counter.transform(test_codes)
    similarities = cosine_similarity(test_matrix, train_matrix)
    test_nls = []
    test_codes_result = []
    for idx, test_simi in tqdm(enumerate(similarities), total=len(similarities)):
        max_idx = find_mixed_nn(test_simi, train_codes, test_codes[idx], bleu_thre, alpha)
        test_nls.append(train_nls[max_idx])
        test_codes_result.append(train_codes[max_idx])
    return test_nls, test_codes_result

import re

def name_convert_to_camel(name):
    name = '_'.join(name.split())
    return re.sub(r'(_[a-z])', lambda x: x.group(1)[1].upper(), name)