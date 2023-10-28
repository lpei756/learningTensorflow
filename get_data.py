# -*- coding: utf-8 -*-
# @Time    : 2023/10/27
# @Author  : Lei
# @Email   : 6222ppt@gmail.com
# @File    : get_data.py
# @Software: PyCharm
# @Brief   : Crawl Google Images

import requests
import re
import os

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.125 Safari/537.36'}
name = input('Please enter the category of images you want the crawler to fetch:')
num = 0
num_1 = 0
num_2 = 0
x = input('Please enter the number of images you want to crawl? (1 equals 60 images, 2 equals 120 images):')
list_1 = []
for i in range(int(x)):
    name_1 = os.getcwd()
    name_2 = os.path.join(name_1, 'data/' + name)
    url = 'https://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word=' + name + '&pn=' + str(i * 30)
    res = requests.get(url, headers=headers)
    htlm_1 = res.content.decode()
    a = re.findall('"objURL":"(.*?)",', htlm_1)
    if not os.path.exists(name_2):
        os.makedirs(name_2)
    for b in a:
        try:
            b_1 = re.findall('https:(.*?)&', b)
            b_2 = ''.join(b_1)
            if b_2 not in list_1:
                num = num + 1
                img = requests.get(b)
                f = open(os.path.join(name_1, 'data/' + name, name + str(num) + '.jpg'), 'ab')
                print('---------Downloading image number' + str(num) + '----------')
                f.write(img.content)
                f.close()
                list_1.append(b_2)
            elif b_2 in list_1:
                num_1 = num_1 + 1
                continue
        except Exception as e:
            print('---------Image number ' + str(num) + 'cannot be downloaded.----------')
            num_2 = num_2 + 1
            continue

print('Download completed, a total of {} images downloaded, {} successfully downloaded, {} duplicated, and {} failed to download.'.format(num + num_1 + num_2, num, num_1, num_2))
