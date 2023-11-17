import os
import re
from tqdm import tqdm
"""
    1、打开images/train
    2、打开train 和 val 
    
    3、将文件的路径写入train.txt
    
"""
path = r'F:\workwork\datasets\XNSJ_VOC\images\train'
# 获取该文件夹下的所有文件名
file_names = []

for root, dirs, files in os.walk(path):
    for file in files:
        file_names.append(file)

# 指定要保存文件名的文本文件路径
output_file = 'F:\workwork\datasets\XNSJ_VOC/train.txt'

# 将文件名写入文本文件
with open(output_file, 'w') as file:
    for file_name in file_names:
        file.write(f'./images/train/{file_name}' + '\n')



