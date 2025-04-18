import json
# import numpy as np
# from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
# from qwen_vl_utils import process_vision_info
# import re
import math


def load_json_file(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data
time_list = ['0.5 seconds', '1.0 seconds','1.5 seconds','2.0 seconds','2.5 seconds','3.0 seconds']
BoxData = load_json_file('./VlmNuscenesData/trackNusceneResultsFull_6points_10000-3500.json')
data = load_json_file('./VlmNuscenesData/trackTrainDataNoTrackFull6points_val.json')
L1 = 0
L2 = 0
L3 = 0
v1 = 0
v2 = 0
v3 = 0
n=0
res = []
for i in range(len(BoxData)):
    box = BoxData[i]
    for rawdata in data:
        if box["sample"] in rawdata["images"][-1]:
            box["sample"] = rawdata['sample_token']
            break
    try:
        response = json.loads(box["response"][0])
        answer = json.loads(box["answer"])
        if  answer["1.0 seconds"]["x"]<=-5000:
            pass
        else:
            L1 += math.sqrt((response["1.0 seconds"]["x"] - answer["1.0 seconds"]["x"]) ** 2 +
                        (response["1.0 seconds"]["y"] - answer["1.0 seconds"]["y"]) ** 2)
        if  answer["2.0 seconds"]["x"]<=-5000:
            pass
        else:
            L2 += math.sqrt((response["2.0 seconds"]["x"] - answer["2.0 seconds"]["x"]) ** 2 +
                            (response["2.0 seconds"]["y"] - answer["2.0 seconds"]["y"]) ** 2)
        if  answer["3.0 seconds"]["x"]<=-5000:
            pass
        else:
            L3 += math.sqrt((response["3.0 seconds"]["x"] - answer["3.0 seconds"]["x"]) ** 2 +
                            (response["3.0 seconds"]["y"] - answer["3.0 seconds"]["y"]) ** 2)
        v1 += abs(response["1.0 seconds"]["velocity"] - answer["1.0 seconds"]["velocity"])
        v2 += abs(response["2.0 seconds"]["velocity"] - answer["2.0 seconds"]["velocity"])
        v3 += abs(response["3.0 seconds"]["velocity"] - answer["3.0 seconds"]["velocity"])

        text = box['response'][0]
        text = eval(text)
        predicted_trajs = [[-text[time_list[j]]["y"], text[time_list[j]]["x"]] for j in range(6)]
        res.append({"sample_token": box["sample"], "predicted_trajs":predicted_trajs})
        n += 1
    except:
        text = box['answer']
        text = eval(text)
        predicted_trajs = [[-text[time_list[j]]["y"], text[time_list[j]]["x"]] for j in range(6)]
        res.append({"sample_token": box["sample"], "predicted_trajs": predicted_trajs})
        print(box["sample"])



L1 = L1/n
L2 = L2/n
L3 = L3/n
v1 = v1/n
v2 = v2/n
v3 = v3/n
print("L1误差为：{}\nL2误差为：{}\nL3误差为：{}\nv1误差为：{}\nv2误差为：{}\nv3误差为：{}\n总样本数为：{}".format(L1, L2, L3, v1, v2, v3, n))
print(len(res))
with open('./VlmNuscenesData/trackResults1_6points.json', 'w') as f:
    json.dump(res, f, indent=4)