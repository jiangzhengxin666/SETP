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

# 数据集设置
BoxData = load_json_file('/data2/jzx/llm_code/VlmNuscenesData/QwenspactialModelNusceneResults_2.json') #spactialModelNusceneResults_2.json
# 结果变量
res10 = 0
res30 = 0
res50 = 0
res100 = 0
m10 = 0.1
m30 = 0
m50 = 0
m100 = 0
error10 =0
error30 =0
error50 = 0
error100 = 0
for box in BoxData:
    try:
        distance = round(float(extract_decimal_numbers(box["answer"])[-1]), 1)
    except:
        continue

    if distance <=0:
        m10+=1
        res10+=box["error"]
        error10 += box["error"] / distance
    elif distance<=20:
        m30+=1
        res30+=box["error"]
        error30 += box["error"] / distance
    elif distance<=50:
        m50+=1
        res50+=box["error"]
        error50 += box["error"] / distance
    else:
        m100+=1
        res100+=box["error"]
        error100 += box["error"] / distance

print("10m内绝对误差:{}。10m内绝对误差百分比：{}%。总案例数：{}".format(round(res10/m10,2), round(100*error10/m10,2), m10))
print("20m内绝对误差:{}。20m内绝对误差百分比：{}%。总案例数：{}".format(round(res30/m30,2), round(100*error30/m30,2), m30))
print("50m内绝对误差:{}。50m内绝对误差百分比：{}%。总案例数：{}".format(round(res50/m50,2), round(100*error50/m50,2), m50))
print("50+m内绝对误差:{}。50+m内绝对误差百分比：{}%。总案例数：{}".format(round(res100/m100,2), round(100*error100/m100,2), m100))