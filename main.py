from sentimentClassifier import SentimentClassifier
import numpy
import csv
content_src = './Sentiment Classification/outdata/totalcontent.txt'
stop_file = './Sentiment Classification/data/stop.txt'
dic_file = './Sentiment Classification/outdata/total-dic.txt'
out = './Sentiment Classification/outdata/totalsplit.txt'
vec_file = './Sentiment Classification/outdata/total.model'
data_file = 'D:/Project/nlp&ml/Sentiment Classification/outdata/total.csv'
model_file = './Sentiment Classification/outdata/test.model'
sentiment = list()
polarity = list()
lines = []
data = csv.reader(open(data_file, 'r', encoding='utf-8'))
for line in data:
    sentiment.append(line[1])
    polarity.append(int(line[2]))
    lines.append(line)
label = numpy.zeros((len(polarity), 3))
i = 0
label = list()
label = numpy.array(label)
m = SentimentClassifier()
m.load_vec(vec_file)
m.load_model(model_file)
print(m.score())