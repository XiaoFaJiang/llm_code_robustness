import pandas as pd
import os
import copy
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


model_names = ["gpt4o", "longcat-13b-base", "longcat-13b-sft", "codeqwen-7b-base", "codeqwen-7b-instruct", 
               "deepseek-v2-coder-16b-instruct", "deepseek-v2-coder-16b-base", "longcat-120b-rl", "qwen2.5-72b-instruct", 
               "deepseek-v2-chat-236b"]

def clean_data(df):
    # 将NaN值填充为0
    df.fillna(0, inplace=True)

    # 将第一列以外的其他所有列置为数值类型
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 剔除origin_pass@1列中值小于等于0.1的行
    df = df[df['origin_pass@1'] > 0.1]

    return df

def get_existed_results():
    ret = {'complete': [], 'instruct': []}
    for path in list(ret.keys()):
        for model in model_names:
            file_path = os.path.join(path, model + '.xlsx')
            if os.path.exists(file_path):
                df = pd.read_excel(file_path)
                df = clean_data(df)  # 清洗数据
                ret[path].append(df)
    return ret

def sort_perturbations():
    realres = get_existed_results()
    perturbations = ["rename", "code_stmt_exchange", "code_expression_exchange", "insert", "code_style"]
    languages = ["python", "cpp", "java", "javascript"]
    results = {per:{} for per in perturbations}
    for perturbation in perturbations: #多种扰动
        res = {lang:[] for lang in languages}
        for lang in languages: #多种语言
            for k,v in realres.items(): # completion和instruct
                for df in v: #一个completion中有多个df(多个model)
                    for i in range(len(df)):
                        if perturbation in df.iloc[i, 0] and lang in df.iloc[i, 0]:
                            pass_drop_values = df['pass-drop@1'].iloc[i]
                            res[lang].append(pass_drop_values)
        results[perturbation] = copy.deepcopy(res)

    perturbation_relation_have_neg(results,perturbations)
    perturbation_relation_no_neg(results,perturbations)
    return results


def perturbation_relation_have_neg(results,perturbations):
    fig, axes = plt.subplots(1, 5, figsize=(10, 8),sharey=True,sharex=True)
    datas = []
    for per in perturbations:
        lst = []
        for k,v in results[per].items():
            lst.extend(v)
        datas.append(np.array(lst))
    
    for i in range(5):
        axes[i].boxplot(datas[i])
        print(f"{' '.join(perturbations[i].split('_')[:2])}",np.mean(datas[i]))
        axes[i].set_title(f"{' '.join(perturbations[i].split('_')[:2])}")
    plt.tight_layout()

    # 显示图表
    plt.show()
    plt.savefig('perturbation_relation(have neg nums).png')


def perturbation_relation_no_neg(results,perturbations):
    fig, axes = plt.subplots(1, 5, figsize=(10, 8),sharey=True,sharex=True)
    datas = []
    for per in perturbations:
        lst = []
        for k,v in results[per].items():
            for num in v:
                if num > 0:
                    lst.append(num)
        datas.append(np.array(lst))
    
    for i in range(5):
        axes[i].boxplot(datas[i])
        axes[i].set_title(f"{' '.join(perturbations[i].split('_')[:2])}")
    plt.tight_layout()

    # 显示图表
    plt.show()
    plt.savefig('perturbation_relation(no neg nums).png')


def model_base_instruct_relation():
    '''
    比较base模型和instruct模型的鲁棒性
    需要看同一系列模型，在同一个数据集下鲁棒性差异
    '''
    base_models = ["longcat-13b-base","codeqwen-7b-base","deepseek-v2-coder-16b-base"]
    instruct_models = ["longcat-13b-sft","codeqwen-7b-instruct","deepseek-v2-coder-16b-instruct"]

    #两种模型在complete数据集下的鲁棒性差异
    res_complete = {'base':[],'instruct':[]}
    base_dfs = []
    instruct_dfs = []
    for model in base_models + instruct_models:
        file_path = os.path.join('complete', model + '.xlsx')
        if os.path.exists(file_path):
            df = pd.read_excel(file_path)
            df = clean_data(df)  # 清洗数据
            if "base" in model:
                base_dfs.append(df)
            else:
                instruct_dfs.append(df)
    
    for base_df in base_dfs:
        res_complete['base'].extend(base_df['pass-drop@1'].to_list())
    
    for instruct_df in instruct_dfs:
        res_complete['instruct'].extend(instruct_df['pass-drop@1'].to_list())


    fig, axes = plt.subplots(1, 2, figsize=(6, 8),sharey=True,sharex=True)
    datas = []
    datas.append(np.array(res_complete['base']))
    datas.append(np.array(res_complete['instruct']))
    for i in range(2):
        axes[i].boxplot(datas[i])
        axes[i].set_title(["base","instruct"][i])
    plt.tight_layout()

    # 显示图表
    plt.show()
    plt.savefig('base_instruct_relation_on_completion_data.png')
    #两种模型在instruct数据集下的鲁棒性差异
    res_instruct = {'base':[],'instruct':[]}
    base_dfs = []
    instruct_dfs = []
    for model in base_models + instruct_models:
        file_path = os.path.join('instruct', model + '.xlsx')
        if os.path.exists(file_path):
            df = pd.read_excel(file_path)
            df = clean_data(df)  # 清洗数据
            if "base" in model:
                base_dfs.append(df)
            else:
                instruct_dfs.append(df)
    
    for base_df in base_dfs:
        res_instruct['base'].extend(base_df['pass-drop@1'].to_list())
    
    for instruct_df in instruct_dfs:
        res_instruct['instruct'].extend(instruct_df['pass-drop@1'].to_list())


    fig, axes = plt.subplots(1, 2, figsize=(6, 8),sharey=True,sharex=True)
    datas = []
    datas.append(np.array(res_instruct['base']))
    datas.append(np.array(res_instruct['instruct']))
    for i in range(2):
        axes[i].boxplot(datas[i])
        axes[i].set_title(["base","instruct"][i])
    plt.tight_layout()

    # 显示图表
    plt.show()
    plt.savefig('base_instruct_relation_on_instruct_data.png')

    #综合差异


    res = {'base':[],'instruct':[]}
    base_dfs = []
    instruct_dfs = []
    for path in ["complete","instruct"]:
        for model in base_models + instruct_models:
            file_path = os.path.join(path, model + '.xlsx')
            if os.path.exists(file_path):
                df = pd.read_excel(file_path)
                df = clean_data(df)  # 清洗数据
                if "base" in model:
                    base_dfs.append(df)
                else:
                    instruct_dfs.append(df)
    
    for base_df in base_dfs:
        res['base'].extend(base_df['pass-drop@1'].to_list())
    
    for instruct_df in instruct_dfs:
        res['instruct'].extend(instruct_df['pass-drop@1'].to_list())


    fig, axes = plt.subplots(1, 2, figsize=(6, 8),sharey=True,sharex=True)
    datas = []
    datas.append(np.array(res['base']))
    datas.append(np.array(res['instruct']))
    for i in range(2):
        axes[i].boxplot(datas[i])
        axes[i].set_title(["base","instruct"][i])
    plt.tight_layout()

    # 显示图表
    plt.show()
    plt.savefig('base_instruct_relation_on_all_data.png')


def performance_robustness_relation():
    '''
    查看performance和robustness是否具有一定相关性
    '''
    results = get_existed_results()
    performance = []
    robustness = []
    for k,v in results.items():
        for df in v:
            performance.extend(df['origin_pass@1'].to_list())
            robustness.extend(df['pass-drop@1'].to_list())
    performance = np.array(performance)
    robustness = np.array(robustness)
    # 计算斯皮尔曼相关系数
    correlation_coefficient, p_value = spearmanr(performance, robustness)

    print(f"斯皮尔曼相关系数: {correlation_coefficient}")
    print(f"p 值: {p_value}")
    # 绘制散点图
    plt.figure()
    plt.scatter(performance, robustness)
    
    # 拟合曲线
    degree = 5  # 拟合的曲线次数，1 表示线性拟合
    coefficients = np.polyfit(performance, robustness, degree)
    poly_func = np.poly1d(coefficients)

    # 生成拟合曲线的 x 和 y 值
    x_fit = np.linspace(performance.min(), robustness.max(), 100)
    y_fit = poly_func(x_fit)

    # 绘制拟合曲线
    plt.plot(x_fit, y_fit, label=f'Fitted Curve (degree={degree})', color='red')
    plt.title('performance_pass@1_relation')
    plt.xlabel('performance')
    plt.ylabel('pass-drop@1')
    
    plt.show()
    plt.savefig('performance_robustness_relation.png')



def all_model_robustness_relation():
    '''
    所有模型robustness比较
    '''
    # 所有模型completion数据robustness比较
    dfs = {}
    for model in model_names:
        file_path = os.path.join('complete', model + '.xlsx')
        if os.path.exists(file_path):
            df = pd.read_excel(file_path)
            df = clean_data(df)  # 清洗数据
            dfs[model] = np.array(df['pass-drop@1'].to_list())

    model_counts = len(list(dfs.keys()))
    fig, axes = plt.subplots(1, model_counts, figsize=(16, 8),sharey=True,sharex=True)
    count = 0
    for k,v in dfs.items():
        axes[count].boxplot(v)
        model_name = k.split('-')
        v_pos = [i for i in v if i >=0]
        print(k,sum(v)/len(v),sum(v_pos)/len(v_pos))
        if len(model_name) > 3:
            axes[count].set_title(model_name[0] + "-" + model_name[-2] + "-" + model_name[-1])
        else:
            axes[count].set_title(k)
        count += 1
    plt.tight_layout()

    # 显示图表
    #plt.show()
    plt.savefig('model_relation_on_completion.png')



    # 所有模型instruct数据robustness比较

    dfs = {}
    for model in model_names:
        file_path = os.path.join('instruct', model + '.xlsx')
        if os.path.exists(file_path):
            df = pd.read_excel(file_path)
            df = clean_data(df)  # 清洗数据
            dfs[model] = np.array(df['pass-drop@1'].to_list())

    model_counts = len(list(dfs.keys()))
    fig, axes = plt.subplots(1, model_counts, figsize=(16, 8),sharey=True,sharex=True)
    count = 0
    for k,v in dfs.items():
        axes[count].boxplot(v)
        model_name = k.split('-')
        if len(model_name) > 3:
            axes[count].set_title(model_name[0] + "-" + model_name[-2] + "-" + model_name[-1])
        else:
            axes[count].set_title(k)
        count += 1
    plt.tight_layout()

    # 显示图表
    #plt.show()
    plt.savefig('model_relation_on_instruct.png')
    # 综合比较

    dfs = {}
    base_models = ["longcat-13b-base","codeqwen-7b-base","deepseek-v2-coder-16b-base"]
    instruct_models = ["longcat-13b-sft","codeqwen-7b-instruct","deepseek-v2-coder-16b-instruct","gpt4o","qwen2.5-72b-instruct"]
    for path in ["complete","instruct"]:
        for model in base_models + instruct_models:
            file_path = os.path.join(path, model + '.xlsx')
            if os.path.exists(file_path):
                df = pd.read_excel(file_path)
                df = clean_data(df)  # 清洗数据
                if dfs.get(model,None):
                    dfs[model].extend(df['pass-drop@1'].to_list())
                else:
                    dfs[model] = df['pass-drop@1'].to_list()

    model_counts = len(list(dfs.keys()))
    fig, axes = plt.subplots(1, model_counts, figsize=(16, 8),sharey=True,sharex=True)
    count = 0
    for k,v in dfs.items():
        axes[count].boxplot(np.array(v))

        v_pos = [i for i in v if i >=0]
        #print(k,sum(v)/len(v),sum(v_pos)/len(v_pos))
        #print(v)
        model_name = k.split('-')
        if len(model_name) > 3:
            axes[count].set_title(model_name[0] + "-" + model_name[-2] + "-" + model_name[-1])
        else:
            axes[count].set_title(k)
        count += 1
    plt.tight_layout()
    # 显示图表
    #plt.show()
    plt.savefig('model_relation_on_all_data.png')


if __name__ == '__main__':
    sort_perturbations()
    #all_model_robustness_relation()
    #performance_robustness_relation()
    #model_base_instruct_relation()
    #sort_perturbations()