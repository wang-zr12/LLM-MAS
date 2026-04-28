import json
import requests
import os
from tqdm import tqdm

# DeepSeek API configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY_1")
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

headers = {
    "Content-Type": "application/json",
    'Accept': 'application/json',
    "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
}

def make_inputs(context):
    dialogues = context.split('\n')
    inputs = []

    for dial in dialogues:
        role = dial.split("：")[0]
        dial = "：".join(dial.split("：")[1:])
        inputs.append({"from": role, "value": dial})
    return inputs


def prepare_messages_for_api(conversations, role, system_message):
    messages = [{"role": "system", "content": system_message}]

    for i in range(len(conversations)):
        sender = conversations[i]['from']
        content = conversations[i]['value']

        if sender == role:
            messages.append({"role": "assistant", "content": f"{sender}：{content}"})
        else:
            messages.append({"role": "user", "content": f"{sender}：{content}"})

    return messages


def get_response_deepseek_api(data):
    context = data['context']
    role = data['role']

    role_information = role_informations[role]
    role_system = f"你是一个角色扮演专家。根据{role_information}扮演{role}进行对话。 "

    conversations = make_inputs(context)
    messages = prepare_messages_for_api(conversations, role, role_system)

    # Make sure the last message is from user, not the role we're simulating
    if messages[-1]["role"] == "assistant":
        # Remove the last message if it's from the assistant
        messages = messages[:-1]
    payload = json.dumps({
        "messages": messages,
        "model": "deepseek-reasoner",  # Use the appropriate model name provided by DeepSeek
        "max_tokens": 512,
    })

    try:
        response = requests.request("POST",DEEPSEEK_API_URL, headers=headers, data=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors

        result = response.json()
        assistant_response = result["choices"][0]["message"]["content"]
        data["model_output"] = assistant_response
        return data
    except Exception as e:
        #print(f"Error calling DeepSeek API: {str(e)}")
        return None



with open('CharacterEval/data/test_data.jsonl', 'r', encoding='utf-8') as f:
    datas = json.load(f)
with open('CharacterEval/data/character_profiles.json', 'r', encoding='utf-8') as f:
    role_informations = json.load(f)

output_file = 'middle_results/generation_sa_r1.jsonl'
batch_size = 2

# 读取已存在的结果数
processed_count = 0
if os.path.exists(output_file):
    with open(output_file, 'r', encoding='utf-8') as f:
        processed_count = sum(1 for _ in f)
else:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

#print(f"Already processed {processed_count} items.")

# 从断点继续处理
results = []
for i, data in tqdm(
        enumerate(datas[processed_count:], start=processed_count),
        total=len(datas),
        initial=processed_count,
        desc="Processing data"):
    result = get_response_deepseek_api(data)
    if result:
        results.append(result)
    else:
        continue
    if (i + 1) % batch_size == 0 or (i + 1) == len(datas):
        with open(output_file, 'a', encoding='utf-8') as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
            f.flush()
        results = []
