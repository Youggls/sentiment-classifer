import csv
import numpy
import jieba

def split(dic_file, content_src, out_file, stop_file):

    content = open(content_src, 'r', encoding='utf-8')
    split_file = open(out_file, 'w', encoding='utf-8')
    stop_list = open(stop_file, 'r', encoding='utf-8')
    text = list()
    stop_word = list()
    for line in stop_list.readlines():
        line=line.strip('\n')
        stop_word.append(line)
    for i in content:
        temp = jieba.cut(i)
        out = list()
        for j in temp:
            if j not in stop_word:
                out.append(j)
                split_file.write(j)
                split_file.write('\n')
        text.append(out)
    return text