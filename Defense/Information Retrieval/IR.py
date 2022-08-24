import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from rouge.rouge import rouge_w_sentence_level, rouge_l_sentence_level

from metrics import compute_metrics


def rouge_w(prediction: str, ground_truth: str):
    gt_tokens = ground_truth.strip().split(" ")
    pre_tokens = prediction.strip().split(" ")
    # ROUGE-W
    recall, precision, f1 = rouge_w_sentence_level(pre_tokens, gt_tokens)
    return recall

# Find the nearest neighbor using cosine simialrity, bleu score, and rouge-w score
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
    # for idx, test_simi in enumerate(similarities):
    for idx, test_simi in tqdm(enumerate(similarities), total=len(similarities)):
        max_idx = find_mixed_nn(test_simi, train_codes, test_codes[idx], bleu_thre, alpha)
        test_nls.append(train_nls[max_idx])
        test_codes_result.append(train_codes[max_idx])
    return test_nls, test_codes_result

df = pd.read_csv('data/Python/train.csv')
train_funcs = df['func_name'].values
train_nls = df['nl'].values

df = pd.read_csv('data/Python/valid.csv')
test_funcs = df['func_name'].values
test_nls = df['nl'].values

# out_funcs, out_nls = nngen(train_nls, train_funcs, test_nls, bleu_thre=9, alpha=0.6) # Java
out_funcs, out_nls = nngen(train_nls, train_funcs, test_nls, bleu_thre=3, alpha=0.1) # Python
df = pd.DataFrame(out_funcs)
df.to_csv("result/Python/valid_new_NNGen.csv", index=False, header=None)
df = pd.DataFrame(out_nls)
df.to_csv("result/Python/valid_new_NNGen_nl.csv", index=False, header=None)
# df = pd.DataFrame(test_funcs)
# df.to_csv("result/Python/test_ref.csv", index=False, header=None)
#
# result = compute_metrics(test_funcs, out_funcs)
# print(result)
