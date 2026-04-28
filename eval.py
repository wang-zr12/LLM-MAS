import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

output_file = f"middle_results/blank_count.xlsx"
eval_file = f"my_results/evaluation_blank.jsonl"
metrics = ["", "Accuracy", "Communication_skills", "Coherence", "Diversity", "Fluency", "Utterance", "Consistency", "Behavior", "Hallucination", "Empathy", "Humanlikeness","Exposure"]
# Define metric groups
metric_groups = {
    "Character Consistency": ["KA", "KE", "KH", "PB", "PU"],
    "Conversational Ability": ["Flu", "Coh", "Cons"],
    "Role-playing Attractiveness": ["HL", "CS", "ED", "Emp"]
}

# Full metric names mapping
full_names = {
    "KA": "Accuracy",
    "KE": "Exposure",
    "KH": "Hallucination",
    "PB": "Behavior",
    "PU": "Utterance",
    "Flu": "Fluency",
    "Coh": "Coherence",
    "Cons": "Consistency",
    "HL": "Humanlikeness",
    "CS": "Communication_skills",
    "ED": "Diversity",
    "Emp": "Empathy"
}

models = ['blank','sa', 'cot', 'sa_r1', 'mas', 'mas_drop_sp', 'mas_drop_cri']

def merge_frame(f1, f2="my_results/角色信息统计结果.xlsx", output_dir="my_results"):
    """
    合并两个Excel文件，匹配相同role的行并计算新指标

    参数:
        f1: 第一个Excel文件路径（包含total_length）
        f2: 第二个Excel文件路径（包含total_length）
        output_dir: 输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 读取数据
    df1 = pd.read_excel(f1)
    df2 = pd.read_excel(f2)

    # 合并数据（基于role列）
    merged = pd.merge(df1, df2, on="role", suffixes=('_f1', '_f2'))

    # 计算新指标
    merged["total_len_eval"] = merged["total_length_f1"] + merged["total_length_f2"]

    # 保存结果
    output_path = os.path.join(output_dir, "sa_emerged.xlsx")
    merged.to_excel(output_path, index=False)
    print(f"结果已保存到 {output_path}")

def content_count(json_data, dict_list=None, ex_labels=None):
    """
    统计JSON数据并返回DataFrame

    参数:
        json_data: 输入的JSON数据
        dict_list: 需要统计长度的字段列表 (如["context", "model_output"])
        labels: 需要额外记录的字段列表

    返回:
        pandas.DataFrame 包含所有统计结果
    """
    rows = []

    if dict_list:
        # 对话/输出数据模式
        for data in json_data:
            row = {}
            total_len = 0

            # 统计指定字段长度
            for field in dict_list:
                if field in data:
                    field_len = len(str(data[field]))
                    row[f"{field}_length"] = field_len
                    total_len += field_len

            row["total_length"] = total_len

            # 记录额外标签字段
            for label in ex_labels:
                if label=="score":
                    row[label] = data[data["metric_en"]]
                else:
                    row[label] = data[label]

            rows.append(row)
    else:
        # 角色属性数据模式
        for role_name, role_data in json_data.items():
            json_str = json.dumps(role_data, ensure_ascii=False)
            rows.append({
                "role": role_name,
                "total_length": len(json_str),
            })

    return pd.DataFrame(rows)

def analyze_character(character_data):
    """分析角色的信息长度和丰富度"""
    results = []
    for role_name, role_data in character_data.items():

        # 计算总信息长度（整个对象的JSON字符串长度）
        json_str = json.dumps(role_data, ensure_ascii=False)
        total_length = len(json_str)

        # 计算叶子节点数量（丰富度）
        richness = 0

        def count_leaves(node):
            nonlocal richness
            if isinstance(node, dict):
                for key, value in node.items():
                    count_leaves(value)
            elif isinstance(node, list):
                for item in node:
                    count_leaves(item)
            else:
                if node is not None:  # 忽略None值
                    richness += 1

        count_leaves(role_data)

        results.append({
            "role": role_name,
            "total_length": total_length,
            "richness": richness
        })

    return pd.DataFrame(results)

def compute_score(json_file):
    score_dict = {}
    with open(json_file, "r", encoding="utf-8") as f:
        records = json.load(f)
    for record in records:
        if record['metric_en'] not in score_dict:
            score_dict[record['metric_en']] = []
        score_dict[record['metric_en']].append(record[record['metric_en']])

    for metric in score_dict:
        print(metric)
        print(sum(score_dict[metric]) / len(score_dict[metric]))
    return score_dict

def correlation(excel_file, label1, label2, metric=None):
    """
    计算两个指标的相关性并绘制拟合图

    参数:
        file1: 第一个Excel文件路径
        label1: 第一个文件中的指标列名
        label2: 第二个文件中的指标列名
        metric: 需要筛选的特定metric_en值（可选）
    """

    # 读取数据
    df1 = pd.read_excel(excel_file)#8033 rows

    # Check if labels exist
    for label in [label1, label2]:
        if label not in df1.columns:
            raise ValueError(f"Column '{label}' not found in DataFrame")

    # 筛选特定metric数据（如果指定）
    if metric:
        if 'metric_en' in df1.columns:
            df1 = df1[df1['metric_en'] == metric]


    # 合并数据（处理可能没有ID列的情况）
    try:
        # 尝试用ID列合并
        merged = pd.merge(df1[label1], df1[label2],on='ID', how='inner')
    except:
        # 如果失败则按索引合并
        merged = pd.concat([df1[[label1]], df1[[label2]]], axis=1).dropna()

    # 检查数据是否为空
    if merged.empty:
        raise ValueError("合并后的数据为空，请检查输入文件或筛选条件")
    # 应用数值变换
    merged[label1] = merged[label1]
    merged[label2] = merged[label2]

    # 计算相关系数
    corr = merged.corr().iloc[0, 1]
    print(f"{metric}相关系数 ({label1} vs {label2}): {corr:.3f}")
    rho = df1[label1].corr(df1[label2], method='spearman')
    print(f"斯皮尔曼相关系数: {rho:.3f}")
    # 绘制拟合图
    plt.figure(figsize=(10, 6))
    sns.regplot(x=label1, y=label2, data=merged,
                scatter_kws={'alpha': 0.5},
                line_kws={'color': 'red'})

    title = f"{label1} vs {label2} (r = {corr:.2f})"
    if metric:
        title += f" | Metric: {metric}"
    plt.title(title)
    plt.xlabel(label1)
    plt.ylabel(label2)
    plt.grid(True)
    plt.show()

def draw_tanc(excel_file, metric, bins=10):
    """
    绘制指标分布柱形图

    参数:
        file: Excel文件路径
        metric: 要分析的指标列名
        bins: 分段数量
    """
    df = pd.read_excel(excel_file)
    data = df[metric].dropna()

    # 自动确定分段区间
    min_val = data.min()
    max_val = data.max()
    intervals = np.linspace(min_val, max_val, bins + 1)

    # 统计各区间的数量
    counts = []
    labels = []
    for i in range(len(intervals) - 1):
        lower = intervals[i]
        upper = intervals[i + 1]
        count = ((data >= lower) & (data < upper)).sum()
        counts.append(count)
        labels.append(f"{lower:.1f}-{upper:.1f}")

    # 最后一个区间包含最大值
    counts[-1] += (data == max_val).sum()

    # 绘制梯形图
    plt.figure(figsize=(12, 6))
    plt.bar(labels, counts, width=0.8,
            color='skyblue', edgecolor='black')

    plt.title(f"{metric} Distribution Statistics (Total: {len(data)} data points)")
    plt.xlabel("Value Range")
    plt.ylabel("Count")
    plt.xticks(rotation=45)

    # 添加数值标签
    for i, v in enumerate(counts):
        plt.text(i, v + 0.5, str(v), ha='center')

    plt.tight_layout()
    plt.show()

def bucket_count(excel_file, metric=None, draw=True):
    df = pd.read_excel(excel_file)

    # Define bucket boundaries
    if metric:
        buckets = [
            (0, 195),
            (196, 362),
            (363, 529),
            (530, 700),
            (700, 900),
            (900, float('inf')),
        ]
    else:
        buckets = [
            (0, 195),
            (196, 362),
            (363, 529),
            (530, 695),
            (696, 862),
            (863, 1029),
            (1030,1362),
            (1363, float('inf')),
        ]
    # 初始化 score 和 bucket_counts
    score = [0] * len(buckets)
    bucket_counts = [0] * len(buckets)

    # Process each row
    for _, row in df.iterrows():
        if metric and row["metric_en"] != metric:
            continue

        total_length = row["total_length"]

        # Determine which bucket the value falls into
        for i, (low, high) in enumerate(buckets):
            if low <= total_length < high:
                score[i] += row["score"]
                bucket_counts[i] += 1
                break

    # 计算归一化得分（避免除以0）
    normalized_score = []
    for s, cnt in zip(score, bucket_counts):
        if cnt > 0:
            normalized_score.append(s / cnt)
        else:
            normalized_score.append(0)  # 如果桶里没有数据，设为0
    if draw:
    # 打印每个桶的数据点数量（用于调试）
        print(f"Bucket Counts({metric}):", bucket_counts)
        title = metric if metric else "Total"
        # 绘制折线图
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(buckets) + 1), normalized_score, marker='o', linestyle='-')
        plt.title(f"Normalized Score by Total Length Bucket ({title})- mas")
        plt.xlabel("Bucket Number")
        plt.ylabel("Average Score")
        plt.xticks(range(1, len(buckets) + 1))
        plt.grid(True)
        plt.show()

    return normalized_score, bucket_counts




def compute_weighted_avg(df, metrics):
    """Compute weighted average for a group of metrics"""
    count_total=0
    score=0
    for i,metric in enumerate(metrics):
        full_metric = full_names.get(metric, metric)
        for _, row in df.iterrows():
            if row["metric_en"] != full_metric:
                continue
            count_total+=1
            score+=row['score']
    avg_score=score/count_total

    return avg_score


def compute_avgs(df):
    results = {}
    for group_name, group_metrics in metric_groups.items():
        results[group_name] = compute_weighted_avg(df, group_metrics)
    return results




def main(c=0, prof=0, score=0, cor=0, l_pic=0, bucket=0, avg=0):
    if c:
        # 读取JSON数据
        with open(eval_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        with open("CharacterEval/data/character_profiles.json", "r", encoding="utf-8") as f:
            cha_data = json.load(f)
        # 统计并生成DataFrame
        df1 = content_count(
            data,
            dict_list=["context", "model_output"],  # 需要统计长度的字段
            ex_labels=["id", "role", "metric_en", "score"]  # 额外需要记录的字段
        )
        df1.to_excel(output_file, index=False)
        print(f"统计结果已保存到 {output_file}")
    if prof:
        df2 = analyze_character(cha_data)
        # 保存为Excel
        df2.to_excel("my_results/角色信息统计结果.xlsx", index=False)
        print("统计结果已保存到 角色信息统计结果.xlsx")
    if score:
        compute_score(eval_file)
    if cor:
        cor_file = "middle_results/sa_emerged.xlsx"
        for metric in metrics:
            correlation(cor_file, "total_len_eval", "score", metric)
    if l_pic:
        tanc_file = "middle_results/mas_count.xlsx"
        draw_tanc(tanc_file, "total_length",)
    if bucket:
        for metric in ["Exposure"]:
            normalized_scores, counts = bucket_count(output_file, metric)
    if avg:
        # Process each model
        all_results = {}

        for model in models:
            excel_file = f'middle_results/{model}_count.xlsx'
            try:
                df = pd.read_excel(excel_file)
                model_results = compute_avgs(df)
                all_results[model] = model_results
            except Exception as e:
                print(f"Error processing {model}: {str(e)}")
                continue

        # Convert results to DataFrame for better visualization
        results_df = pd.DataFrame.from_dict(all_results, orient='index')
        print(results_df)

        # Save results to Excel
        results_df.to_excel("model_metric_analysis.xlsx")

if __name__ == "__main__":

    #merge_frame(output_file)
    main(0,0,0, 0, 0, avg=1)
