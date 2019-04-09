from gensim.models import word2vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import random
import numpy
import csv
import pkuseg
from toVec import onehot
content_src = './Sentiment Classification/outdata/Elong-content.txt'
stop_file = './Sentiment Classification/data/stop.txt'
dic_file = './Sentiment Classification/outdata/Elong-dic.txt'
out = './Sentiment Classification/outdata/Elong-split.txt'
model_file = './Sentiment Classification/outdata/make-up.model'
data_file = 'D:/Project/nlp&ml/Sentiment Classification/outdata/total.csv'

def predict(lr1, lr2, lr3, X):
    p1 = lr1.predict_proba([X])[0][1]
    p2 = lr2.predict_proba([X])[0][1]
    p3 = lr3.predict_proba([X])[0][1]

    if p1 >= p2 and p1 >= p3:
        return numpy.array([1, 0, 0])
    elif p2 >= p1 and p2 >= p3:
        return numpy.array([0, 1, 0])
    elif p3 >= p1 and p3 >= p2:
        return numpy.array([0, 0, 1])
    else:
        return numpy.array([0, 0, 0])

sentiment = list()
polarity = list()
sentence = list()
f = csv.reader(open(data_file, 'r', encoding='utf-8'))

count = 0
for line in f:
    count += 1
    sentiment.append(line[1])
    polarity.append(int(line[2]))

polarity = numpy.array(polarity)
# model = word2vec.Word2Vec(sentiment, min_count = 2)
# model.save(model_file)

model = onehot(sentiment)

label = numpy.zeros((len(polarity), 3))

i = 0
X = list()
label = list()
for p in polarity:
    if p == 1:
        if random.random() <= 0.3:
            X.append(model[sentiment[i]])
            label.append([0, 0, 1])
    elif p == 0:
        X.append(model[sentiment[i]])
        label.append([0, 1, 0])
    elif p == -1:
        X.append(model[sentiment[i]])
        label.append([1, 0, 0])
    
    i+=1
X = numpy.array(X)
label = numpy.array(label)
# print(label)
lr_pos = SVC()
lr_neu = SVC()
lr_neg = SVC()
X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.2, random_state=33)
print(X_train[1])
lr_pos.fit(X_train, y_train[:, 2])
lr_neu.fit(X_train, y_train[:, 1])
lr_neg.fit(X_train, y_train[:, 0])

TP = 0
FP = 0
FN = 0
TN = 0
pred = numpy.zeros((len(X_test), 3))

for i in range(len(X_test)):
    pred[i] = predict(lr1=lr_neg, lr2=lr_neu, lr3=lr_pos, X=X_test[i])

pos = 0
neg = 0
neu = 0
for i in range(len(y_test)):
    if y_test[i][2] == 1:
        pos += 1
    elif y_test[i][1] == 1:
        neu += 1
    elif y_test[i][0] == 1:
        neg += 1

print('Positive = {} Neutral = {} Negative = {}'.format(pos, neu, neg))

class_type = 1
for index in range(len(X_test)):
    if pred[index][class_type + 1] == 1 and y_test[index][class_type + 1] == 1:
        TP += 1
    elif pred[index][class_type + 1] == 1 and y_test[index][class_type + 1] == 0:
        FP += 1
    elif pred[index][class_type + 1] == 0 and y_test[index][class_type + 1] == 1:
        FN += 1
    elif pred[index][class_type + 1] == 0 and y_test[index][class_type + 1] == 0:
        TN += 1

print('TP = {}, FP = {}, FN = {}, TN = {}'.format(TP, FP, FN, TN))
print('accuracy rate: ', (TP + TN) / len(y_test))
print('precision rate: ', TP / (TP + FP))
print('recall rate: ', TP / (TP + FN))
print('The F1 is: ', 2 * TP / (2 * TP + FP + FN))