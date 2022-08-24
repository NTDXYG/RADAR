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

def find_mixed_nn(simi, diffs, test_diff, idx, bleu_thre, alpha):
    """Find the nearest neighbor using cosine simialrity and bleu score"""
    candidates = simi.argsort()[-bleu_thre:][::-1]
    max_score = 0
    max_idx = 0
    for j in candidates:
        if(j != idx):
            bleu_score = sentence_bleu([diffs[j].split()], test_diff.split())
            # jaccard_score = Jaccard(diffs[j], test_diff)
            rouge_score = rouge_w(diffs[j], test_diff)
            score = alpha*bleu_score + (1-alpha)*rouge_score
            if score > max_score:
                max_score = score
                max_idx = j
    return max_idx

def nngen(train_codes, train_nls, test_codes, start, bleu_thre, alpha):
    counter = TfidfVectorizer()
    train_matrix = counter.fit_transform(train_codes)
    test_matrix = counter.transform(test_codes)
    similarities = cosine_similarity(test_matrix, train_matrix)
    test_nls = []
    test_codes_result = []
    # for idx, test_simi in enumerate(similarities):
    for idx, test_simi in enumerate(similarities):
        max_idx = find_mixed_nn(test_simi, train_codes, test_codes[idx], start+idx, bleu_thre, alpha)
        test_nls.append(train_nls[max_idx])
        test_codes_result.append(train_codes[max_idx])
    return test_nls, test_codes_result

df = pd.read_csv('data/Python/train.csv')
train_funcs = df['func_name'].values
train_nls = df['nl'].values

df = pd.read_csv('data/Python/train.csv')
test_funcs = df['func_name'].values
test_nls = df['nl'].values

out_funcs = []
out_nls = []
for start in tqdm(range(0,len(test_nls), 4000), total=len(train_nls)/4000):
    out_func, out_nl = nngen(train_nls, train_funcs, test_nls[start: start + 4000], start, bleu_thre=3, alpha=0.1)
    out_funcs.extend(out_func)
    out_nls.extend(out_nl)
df = pd.DataFrame(out_funcs)
df.to_csv("result/Python/train_new_NNGen.csv", index=False, header=None)
df = pd.DataFrame(out_nls)
df.to_csv("result/Python/train_new_NNGen_nl.csv", index=False, header=None)
# df = pd.DataFrame(test_funcs)
# df.to_csv("result/Java/valid_ref.csv", index=False, header=None)

# result = compute_metrics(test_funcs, out_funcs)
# print(result)
