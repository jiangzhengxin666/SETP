import json
import numpy as np
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import re
import math

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
model_dir = '/data2/jzx/LLaMA-Factory/models/qwen2_vl_2b_trackTrainSimple_6points_10000-3500'
# model_dir = '/data2/jzx/LLaMA-Factory/models/qwen2_vl_2b_trackTrain3-3500'
# model_dir = '/data2/jzx/LLaMA-Factory/models/qwen2_vl_2b_trackTrain-1500'
# model_dir = '/data2/jzx/LLaMA-Factory/models/qwen2_vl_2b_trackTrain-simple-2'
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_dir, torch_dtype="auto", device_map="auto" # cuda:1
)
min_pixels = 256*28*28
max_pixels = 1600*900*7
processor = AutoProcessor.from_pretrained(model_dir, min_pixels=min_pixels, max_pixels=max_pixels)

# 数据集设置
BoxData = load_json_file('./VlmNuscenesData/trackTrainDataNoTrackFull6points_val.json') # trackTrainDataSimple_val.json # trackTrainData_val.json
# 结果变量
res = []
m = 0
n = 0
L2error_0_5 = 0
L2error_1 = 0
L2error_1_5 = 0
L2error_2 = 0
L2error_2_5 = 0
L2error_3 = 0
print(len(BoxData))
for box in BoxData:
    m+=1
    res.append({"sample":"", "response":"", "answer":""})
    # 添加system
    system = box["system"]

    # 问题
    question =  box["messages"][0]["content"]
    res[-1]["sample"] = box["sample_token"]

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
                    "image": box["images"][0],
                    "imag1": box["images"][1],
                    "imag2": box["images"][2],
                    "imag3": box["images"][3],
                    "imag4": box["images"][4],
                    "imag5": box["images"][5],
                    #"imag6": box["images"][6]
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

    answers = json.loads(box["messages"][1]["content"])

    try:
        print("###########")
        response = json.loads(output_text[0])
        if  answers["0.5 seconds"]["x"] < -1000:
            raise Exception
        L2error_0_5 += math.sqrt((response["0.5 seconds"]["x"] - answers["0.5 seconds"]["x"]) ** 2 +
                                               (response["0.5 seconds"]["y"] - answers["0.5 seconds"]["y"]) ** 2)
        n += 1
        if  answers["1.0 seconds"]["x"] < -1000:
            raise Exception
        L2error_1 += math.sqrt((response["1.0 seconds"]["x"] - answers["1.0 seconds"]["x"]) ** 2 +
                                               (response["1.0 seconds"]["y"] - answers["1.0 seconds"]["y"]) ** 2)
        if  answers["1.5 seconds"]["x"] < -1000:
            raise Exception
        L2error_1_5 += math.sqrt((response["1.5 seconds"]["x"] - answers["1.5 seconds"]["x"]) ** 2 +
                                               (response["1.5 seconds"]["y"] - answers["1.5 seconds"]["y"]) ** 2)
        if  answers["2.0 seconds"]["x"] < -1000:
            raise Exception
        L2error_2 += math.sqrt((response["2.0 seconds"]["x"] - answers["2.0 seconds"]["x"]) ** 2 +
                                               (response["2.0 seconds"]["y"] - answers["2.0 seconds"]["y"]) ** 2)
        if  answers["2.5 seconds"]["x"] < -1000:
            raise Exception
        L2error_2_5 += math.sqrt((response["2.5 seconds"]["x"] - answers["2.5 seconds"]["x"]) ** 2 +
                                               (response["2.5 seconds"]["y"] - answers["2.5 seconds"]["y"]) ** 2)
        if  answers["3.0 seconds"]["x"] < -1000:
            raise Exception
        L2error_3 += math.sqrt((response["3.0 seconds"]["x"] - answers["3.0 seconds"]["x"]) ** 2 +
                                               (response["3.0 seconds"]["y"] - answers["3.0 seconds"]["y"]) ** 2)

        #n += 1
        print("0.5秒L2误差为{}。".format(L2error_0_5 / n))
        print("1秒L2误差为{}。".format(L2error_1/n))
        print("1.5秒L2误差为{}。".format(L2error_1_5 / n))
        print("2秒L2误差为{}。".format(L2error_2/n))
        print("2.5秒L2误差为{}。".format(L2error_2_5 / n))
        print("3秒L2误差为{}。".format(L2error_3/n))
        print("规划成功率为{}%。".format(100 * round(n / m, 1)))
    except:
        print("模型输出结果:")
        print(output_text)
        print("实测结果:")
        print(answers)
    # print(response)
    res[-1]["response"] = output_text

    # 答案
    answers = box["messages"][1]["content"]
    res[-1]["answer"] = answers

    print(m)

print("0.5秒L2误差为{}。".format(L2error_0_5 / n))
print("1秒L2误差为{}。".format(L2error_1 / n))
print("1.5秒L2误差为{}。".format(L2error_1_5 / n))
print("2秒L2误差为{}。".format(L2error_2 / n))
print("2.5秒L2误差为{}。".format(L2error_2_5 / n))
print("3秒L2误差为{}。".format(L2error_3 / n))
print("规划成功率为{}%。".format(100 * round(n / m, 6)))

with open('./VlmNuscenesData/trackNusceneResultsFull_6points2_10000-3500.json', 'w') as f:
    json.dump(res, f, indent=4)