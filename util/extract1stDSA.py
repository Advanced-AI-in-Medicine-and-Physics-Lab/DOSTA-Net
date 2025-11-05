#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File        :extract1stDSA.py
@Description : extract all the first DSA image from the video
@Time        :2024/12/09 22:06:02
@Author      :Jinkui Hao
@Version     :1.0
'''
import os
import shutil
from tqdm import tqdm

videoPath = 'data/Private/OriAll'
targetPath = 'data/Private/firstFrame'
os.makedirs(targetPath,exist_ok=True)
for item in tqdm(os.listdir(videoPath)):
    imgP= os.path.join(videoPath,item)
    imgList = os.listdir(imgP)
    imgList.sort()
    if len(imgList) >0:
        source = os.path.join(videoPath, item, imgList[0])
        target = os.path.join(targetPath, item+imgList[0])
        shutil.copy(source, target)


