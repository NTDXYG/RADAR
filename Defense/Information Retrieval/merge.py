import pandas as pd

df = pd.read_csv("data/Python/test.csv")
nl_list = df['nl'].tolist()
func_name_list = df['func_name'].tolist()

df = pd.read_csv("result/Python/test_new_NNGen.csv", header=None)
ir_list = df[0].tolist()

df = pd.read_csv("result/Python/test_new_NNGen_nl.csv", header=None)
ir_nl_list = df[0].tolist()

data_list = []
for i in range(len(nl_list)):
    data_list.append(['<e> NL: ' + ir_nl_list[i] + ' name: ' + ir_list[i] + ' </e> ' + nl_list[i], func_name_list[i]])
    # data_list.append([nl_list[i] + " " + ir_list[i], func_name_list[i]])

df = pd.DataFrame(data_list, columns=['nl', 'func_name'])
df.to_csv("data/Python/test_ir(1).csv", index=False)
