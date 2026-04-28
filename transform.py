import json
import copy

id2metric_file = f'CharacterEval/data/id2metric.jsonl'
input_file = f'middle_results/generation_mas_1.jsonl'
output_file_final = f"my_results/generation_trans_mas_1_.jsonl"
ab_id_file = f'my_results/ab_ids.txt'


def blank_file(input_file="my_results/generation_trans_blank.jsonl", output="test/score_stability_test.jsonl"):
    with open(input_file, 'r', encoding='utf-8') as f:
        datas = json.load(f)
    i = 0
    results = []
    for data in datas:
        if i < 7800:
            i +=1
            continue
        results.append(data)

    f = open(output, 'a', encoding='utf-8')
    f.write(json.dumps(results, ensure_ascii=False, indent=4))
    f.close()


def find_abnormal_model_outputs(input_file, output_id_file=None):
    empty_ids = []
    short_outputs = []  # 存储短输出及其ID
    empty_count = 0
    short_count = 0

    with open(input_file, 'r', encoding='utf-8') as fin:
        datas = json.load(fin)

    for data in datas:
        model_output = data['model_output']
        if model_output is None or not str(model_output).strip():
            empty_ids.append(data['id'])
            empty_count += 1
        elif len(model_output) < 5:  # 检查长度小于5的非空输出
            short_outputs.append((data['id'], model_output))
            short_count += 1

    # 写入结果文件
    if output_id_file:
        with open(output_id_file, 'w', encoding='utf-8') as fout:
            # 写入空输出的ID
            fout.write("=== Empty Outputs ===\n")
            for eid in empty_ids:
                fout.write(f"{eid}\n")

            # 写入短输出的ID和内容
            fout.write("\n=== Short Outputs (length < 5) ===\n")
            for sid, output in short_outputs:
                fout.write(f"{sid}: {output}\n")

    print(f"Empty Count: {empty_count}")
    print(f"Short Output Count: {short_count}")
    print(f"Total Abnormal Count: {empty_count + short_count}")


def construct_format(id2metric_file, input_file, output_file_final):
    with open(id2metric_file, 'r', encoding='utf-8') as f:
        id_metric = json.load(f)
    with open(input_file, 'r', encoding='utf-8') as f:
        datas = json.load(f)

    results = []
    for data in datas:
        if data['model_output'] is not None and data['model_output'] != "ERROR":
            model_output = data['model_output'].split("\n")[0]  # Prevent continuous generation
            data['model_output'] = model_output
            if str(data['id']) in id_metric:
                for x in id_metric[str(data['id'])]:
                    data['metric_en'] = x[0]
                    data['metric_zh'] = x[1]
                    tmp = copy.deepcopy(data)
                    results.append(tmp)

    f = open(output_file_final, 'w', encoding='utf-8')
    f.write(json.dumps(results, ensure_ascii=False, indent=4))
    f.close()


def clean_jsonl(input_file, output_file=None):
    count = 0
    with open(input_file, 'r', encoding='utf-8') as fin:
        datas = json.load(fin)

    results = []
    for data in datas:
        model_output = data['model_output']
        role = data['role']

        # 删除开头的所有换行符
        cleaned_output = model_output.lstrip('\n').strip()

        # 记录空输出
        if not cleaned_output:
            count += 1

        # 检查开头是否包含角色名（简单匹配方式）
        if not cleaned_output.startswith(f"{role}：") and not cleaned_output.startswith(f"{role}:"):
            cleaned_output = f"{role}：{cleaned_output}"

        # 更新数据
        data["model_output"] = cleaned_output
        results.append(data)
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as fout:
            fout.write(json.dumps(results, ensure_ascii=False) + "\n")
    else:
        with open(input_file, 'w', encoding='utf-8') as fout:
            fout.write(json.dumps(results, ensure_ascii=False) + "\n")

    print(f"Empty Count:{count}")
    return count


def fix_jsonl_file(input_path, output_path=None):
    if output_path is None:
        with open(input_path, 'r', encoding='utf-8') as f:
            # 保留非空行且不去除行末换行符
            lines = [line for line in f if line.strip()]

        # 若要重新拼接为字符串
        content = ''.join(lines)
        fixed_content = content.replace('}\n{', '},\n{')
        fixed_content = "[\n" + fixed_content + "\n]"
        with open(input_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
    else:
        # 输出到新文件
        with open(input_path, 'r', encoding='utf-8') as fin, \
                open(output_path, 'w', encoding='utf-8') as fout:
            content = fin.read()
            fixed_content = content.replace('}\n{', '},\n{')
            fixed_content = "[\n" + fixed_content + "\n]"
            fout.write(fixed_content)


def main():
    #fix_jsonl_file(input_file)
    #c = clean_jsonl(input_file)
    construct_format(id2metric_file, input_file, output_file_final)
    if c:
        find_abnormal_model_outputs(output_file_final, None)

if __name__ == "__main__":
    main()