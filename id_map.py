import json

ids = set()
with open('middle_results/generation_mas_1.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        if isinstance(data, dict) and 'id' in data:  # Case 1: data is a dict
            ids.add(data['id'])
        elif isinstance(data, list):  # Case 2: data is a list of dicts
            for item in data:
                if isinstance(item, dict) and 'id' in item:
                    ids.add(item['id'])

output_data = {}
for i, _id in enumerate(ids):
    if i < 500:
        output_data[_id] = [
            ["Accuracy", "知识准确率"],
            ["Hallucination", "知识幻觉性"],
            ["Behavior", "行为一致性"],
            ["Coherence", "对话连贯性"],
            ["Consistency", "对话一致性"],
            ["Communication_skills", "交流技巧"]
        ]
    else:
        output_data[_id] = [
            ["Exposure", "知识曝光度"],
            ["Humanlikeness", "类人程度"],
            ["Empathy", "共情度"],
            ["Utterance", "言语一致性"],
            ["Fluency", "对话流利度"],
            ["Diversity", "表现多样性"]
        ]

# Write to JSON file
with open('CharacterEval/data/id2metric_.jsonl', 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)