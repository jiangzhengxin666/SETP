import json
import numpy as np
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import re

def extract_decimal_numbers(text):
    # 正则表达式匹配带小数点的数字
    pattern = r'\d+\.\d+'
    res =  re.findall(pattern, text[-1])
    return res

def load_json_file(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

# 设置模型
model_dir = '/data2/jzx/LLaMA-Factory/models/qwen2_vl_lora_sft'
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_dir, torch_dtype="auto", device_map="auto" # cuda:1
)
min_pixels = 256*28*28
max_pixels = 1600*900*2
for i in range(10000000):
    try:
        processor = AutoProcessor.from_pretrained(model_dir, min_pixels=min_pixels, max_pixels=max_pixels)
        break
    except:
        continue

# 数据集设置
BoxData = load_json_file('/data2/jzx/llm_code/VlmNuscenesData/boundingBoxObjectNuscene_2.json')
# 结果变量
res = []
m=1
for box in BoxData:
    m+=1
    res.append({"question":"", "response":"", "answer":""})
    # 添加system
    system = box["system"]

    # 问题
    name = box["messages"][1]["content"].split(' ')[3]
    question =  box["messages"][0]["content"]
    res[-1]["question"] = question

    # 模型回答
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": system
                 }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": box["images"],
                },
                {"type": "text", "text": question
                 }
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    try:
        response = round(float(extract_decimal_numbers(output_text)[-1]),1)
    except:
        response = -1
    # print(response)
    res[-1]["response"] = output_text

    # 答案
    answers = name
    res[-1]["answer"] = answers


    print(output_text)
    print(answers, m)

with open('VlmNuscenesData/objDetectNusceneResults_2.json', 'w') as f:
    json.dump(res, f, indent=4)