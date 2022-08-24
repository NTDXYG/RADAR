import re

import torch
from sko.tools import set_run_mode
from tqdm import tqdm
from transformers import RobertaTokenizerFast, T5ForConditionalGeneration, PLBartTokenizer, PLBartForConditionalGeneration, GPT2TokenizerFast, GPT2LMHeadModel

from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from string import digits, ascii_lowercase

# pretrain the word2vec
df = pd.read_csv("java_func/train_ir.csv")
data_list = df['func_name'].tolist()
df = pd.read_csv("java_func/valid_ir.csv")
data_list.extend(df['func_name'].tolist())
df = pd.read_csv("java_func/test_ir.csv")
data_list.extend(df['func_name'].tolist())
data_list = [data.split() for data in data_list]

model = Word2Vec(sentences=data_list, vector_size=300, window=5, min_count=1, workers=4)
model.save("java_word2vec.model")

from codebleu import compute_codebleu

model = Word2Vec.load("java_word2vec.model")
df = pd.read_csv("java_func/test_ir.csv")
func_name_list = df['func_name'].tolist()
df = pd.read_csv("JSCG/test_gen.csv")
nl_list, code_list = df['nl'].tolist(), df['code'].tolist()

func_name_attack_list = []
import random

codet5_model = T5ForConditionalGeneration.from_pretrained('/home/yangguang/models/codet5-base')
codet5_model.load_state_dict(torch.load('/home/yangguang/PycharmProjects/CodeGenPython/Encoder_Decoder/JSCG/codet5/valid_output/checkpoint-best-bleu/pytorch_model.bin'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = PLBartTokenizer.from_pretrained('/home/yangguang/models/codet5-base')
codet5_model.to(device)

def get_score(x, y):
    input_ids = tokenizer(x ,return_tensors="pt", max_length=100, padding="max_length", truncation=True)
    summary_text_ids = codet5_model.generate(
        input_ids=input_ids["input_ids"].to(device),
        attention_mask=input_ids["attention_mask"].to(device),
        bos_token_id=codet5_model.config.bos_token_id,
        eos_token_id=codet5_model.config.eos_token_id,
        max_length=256,
        num_beams=10,
    )
    gen = tokenizer.decode(summary_text_ids[0], skip_special_tokens=True)
    score = compute_codebleu([y], [gen])
    return score


from sko.GA import GA

def to_lower_camle_case(x):
    s = re.sub('_([a-zA-Z])', lambda m: (m.group(1).upper()), x)
    return s[0].lower() + s[1:]

result_list = []

def gen_garble_code():
    list1 = []
    for i in digits:
        list1.append(i)  # 得到字母字符并放入列表
    for i in ascii_lowercase:
        list1.append(i)  # 得到数字字符并放入列表
    k_words = random.randint(2, 6)
    result_list = []
    result_list.append(''.join(random.choices(list1, k=k_words)))
    return ' '.join(result_list)

def gen_random_delete_char(word):
    temp = random.randint(0, len(word))
    return word[:temp] + word[temp+1:]

def gen_random_swap_char(word):
    if(len(word)>2 and word != len(word) * word[0]):
        try:
            temp = random.randint(0, len(word)-1)
            return word[:temp] + word[temp+1] + word[temp] + word[temp+2:]
        except:
            return word
    else:
        return word

def gen_vis_simi_replace(word):
    result = word
    for w in word:
        if(w == '2'):
            result.replace(w, 'to')
        if (w == '4'):
            result.replace(w, 'for')
        if (w == 'l'):
            result.replace(w, '1')
        if (w == 'o'):
            result.replace(w, '0')
        if (w == 'q'):
            result.replace(w, '9')
        if (w == 's'):
            result.replace(w, '5')
    return result

for index in tqdm(range(len(func_name_list))):
    # print(get_score(nl_list[index], code_list[index]))
    data = func_name_list[index]
    temp_list = []
    lis = [[] for i in range(len(data.split()))]
    for i in range(len(data.split())):
        sims = model.wv.most_similar(data.split()[i], topn=5)  # get other similar words
        for k in sims:
            lis[i].append(k[0])
        lis[i].append(gen_random_delete_char(data.split()[i]))
        lis[i].append(gen_random_swap_char(data.split()[i]))
        lis[i].append(gen_vis_simi_replace(data.split()[i]))
    func = to_lower_camle_case('_'.join(data.split()))

    count = 0
    def schaffer(p):
        attack = []
        for i in range(len(p)):
            attack.append(lis[i][int(p[i])])
        x_attack = to_lower_camle_case('_'.join(attack))
        nl = nl_list[index]
        nl = nl.replace(func, x_attack)
        code = code_list[index]
        score = get_score(nl, code)
        return score

    set_run_mode(schaffer, 'cached')
    ga = GA(func=schaffer, n_dim=len(data.split()), size_pop=20, max_iter=50, prob_mut=0.001, lb=[0 for i in range(len(data.split()))], ub=[7 for i in range(len(data.split()))], precision=1, early_stop=3)
    # ga.to(device=device)
    best_x, best_y = ga.run()
    for i in range(len(best_x)):
        temp_list.append(lis[i][int(best_x[i])])
    nl = nl_list[index]
    print('------------------')
    print(get_score(nl_list[index], code_list[index]))
    print(func, '---------', to_lower_camle_case('_'.join(temp_list)))
    print(best_y[0])
    print('------------------')
    if(best_y[0]<get_score(nl_list[index], code_list[index])):
        nl = nl.replace(func, to_lower_camle_case('_'.join(temp_list)))
    code = code_list[index]
    # print('best_x:', best_x, '\n', 'best_y:', best_y)
    result_list.append([nl, code])

df = pd.DataFrame(result_list, columns=['nl', 'code'])
df.to_csv("JSCG/test_codet5_attack.csv", index=False)
