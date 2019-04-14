import csv
import numpy
import pkuseg
from gensim.models import word2vec

content_src = './Sentiment Classification/outdata/totalcontent.txt'
dic_file = './Sentiment Classification/outdata/total-dic.txt'
out = './Sentiment Classification/outdata/totalsplit.txt'
stop_file = './Sentiment Classification/data/stop.txt'
model_file = './Sentiment Classification/outdata/total.model'

f = open(out, 'r', encoding='utf-8')
sentence = []
stop_words = []
stop = open(stop_file, 'r', encoding='utf-8')
for word in stop.readlines():
    stop_words.append(word.strip('\n'))

count = 0
for line in f.readlines():
    temp = line.strip('\n').split(' ')
    out = []
    for word in temp:
        if word not in stop_words:
            out.append(word)
    sentence.append(out)
    
    if count % 10000 == 0:
        print(out)
    count += 1

model = word2vec.Word2Vec(sentence, min_count=3, workers=8)
model.save(model_file)
print(model.most_similar(['很好']))