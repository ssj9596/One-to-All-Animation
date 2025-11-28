import os
import numpy as np
import pandas as pd
from pandarallel import pandarallel
import regex as re
from collections import Counter
from tqdm import tqdm

def find_videos(text):
    # 定义正则表达式模式，匹配 "the video" 后面紧跟的一个单词或者逗号
    pattern = r'\bthe\s+video\s*(\w+|,)'
    # 使用正则表达式查找所有匹配的模式
    matches = re.findall(pattern, text, re.IGNORECASE)
    # 准备结果列表
    results_after = [match for match in matches]

    pattern = r'(?:\b\w+|,)?\s*the\s+video'
    matches_with_context = re.findall(pattern, text, re.IGNORECASE)
    results_before = []
    for match in matches_with_context:
        if match:
            # 分割匹配结果以提取前面的单词或逗号
            # 由于我们使用了非捕获组，所以 match 可能包含空字符串
            preceding_part = match.split(' the video')[0].strip()
            # 添加 "the video" 到结果中
            results_before.append(preceding_part)

    pattern = r'\bthe\s+image\s*(\w+|,)'
    # 使用正则表达式查找所有匹配的模式
    matches = re.findall(pattern, text, re.IGNORECASE)
    # 准备结果列表
    results_image_after = [match for match in matches]
    
    pattern = r'(?:\b\w+|,)?\s*the\s+image'
    matches_with_context = re.findall(pattern, text, re.IGNORECASE)
    results_image_before = []
    for match in matches_with_context:
        if match:
            # 分割匹配结果以提取前面的单词或逗号
            # 由于我们使用了非捕获组，所以 match 可能包含空字符串
            preceding_part = match.split(' the image')[0].strip()
            # 添加 "the video" 到结果中
            results_image_before.append(preceding_part)

    return results_after, results_before, results_image_after, results_image_before


def find_preceding_words_or_commas(text):
    # 定义正则表达式模式，匹配 "the video" 前面的单词或逗号
    pattern = r'(?:\b\w+|,)?\s*the\s+video'

    # 使用 regex 模块的 findall 方法查找所有匹配的模式
    # regex 模块的 findall 默认情况下是支持非贪婪匹配的
    matches_with_context = re.findall(pattern, text, re.IGNORECASE)

    # 准备结果列表
    results = []

    # 遍历匹配结果，提取 "the video" 及其前面的部分
    for match in matches_with_context:
        if match:
            # 分割匹配结果以提取前面的单词或逗号
            # 由于我们使用了非捕获组，所以 match 可能包含空字符串
            preceding_part = match.split(' the video')[0].strip()
            # 添加 "the video" 到结果中
            results.append(preceding_part + ' the video')

    return results

# # 示例文本
pandarallel.initialize(nb_workers=10, progress_bar=True)
df = pd.read_csv("../training_data_0814_with_sam10M.csv")
ret = df['text'].parallel_apply(
    find_videos
)
# df['prompt_to_clean'].to_csv("prompt_to_clean.csv", index=False)

# 应用函数并打印结果
# for text in text_examples:
    # print(find_preceding_words_or_commas(text))

import ast
def count_func(df):
    counts = Counter()
    for i in tqdm(range(len(df))):
        nouns = df.iloc[i]['prompt_to_clean']
        nouns = set(ast.literal_eval(nouns))
        counts.update(nouns)
    return counts

import ipdb
ipdb.set_trace()
df['video_after'], df['video_before'], df['image_after'], df['image_before'] = zip(*ret)
df = pd.read_csv("prompt_to_clean.csv")
counts = count_func(df)
import ipdb
ipdb.set_trace()

LLAVA_PREFIX = [
    "The video shows",
    "The video captures",
    "The video features",
    "The video depicts",
    "The video presents",
    "The video features",
    "The video is ",
    "In the video,",
    "The image shows",
    "The image captures",
    "The image features",
    "The image depicts",
    "The image presents",
    "The image features",
    "The image is ",
    "The image portrays",
    "In the image,",
]


'the video,'
'the video is'
'the video progresses'
'the video shows'
'the video includes'
'the video frames'
'the video features'
'the video displays'
'the video showcases'
'the video depicts'
'In the video'
'throughout the video'
'As the video progresses'