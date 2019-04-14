from gensim.models import word2vec
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.externals import joblib
import random
import numpy
import csv
import pkuseg
from toVec import onehot
import pickle
def save(file_path, model):
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)
def load(file_path):
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
        return model
content_src = './Sentiment Classification/outdata/totalcontent.txt'
stop_file = './Sentiment Classification/data/stop.txt'
dic_file = './Sentiment Classification/outdata/total-dic.txt'
out = './Sentiment Classification/outdata/totalsplit.txt'
model_file = './Sentiment Classification/outdata/total.model'
data_file = 'D:/Project/nlp&ml/Sentiment Classification/outdata/total.csv'

model = word2vec.Word2Vec.load(model_file)
sentiment = list()
polarity = list()
data = csv.reader(open(data_file, 'r', encoding='utf-8'))
model = word2vec.Word2Vec.load(model_file)
for line in data:
    if line[1] in model:
        sentiment.append(line[1])
        polarity.append(int(line[2]))

label = numpy.zeros((len(polarity), 3))

i = 0
X = list()
label = list()
for p in polarity:
    X.append(model[sentiment[i]])
    label.append(p)
    i+=1
X = numpy.array(X)
label = numpy.array(label)
lr = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.2, random_state=33)

X_train_ = []
y_train_ = []
for x, y in zip(X_train, y_train):
    if y == 1 and random.random() < 0.75:
        X_train_.append(x)
        y_train_.append(y)
    elif y != 1:
        X_train_.append(x)
        y_train_.append(y)
X_train_ = numpy.array(X_train_)
y_train_ = numpy.array(y_train_)
pos = 0
neu = 0
neg = 0
for i in y_train_:
    if i == 1:
        pos += 1
    elif i == 0:
        neu += 1
    elif i == -1:
        neg += 1
print('POS = {}, NEU = {}, NEG = {}'.format(pos, neu, neg))
lr.fit(X_train_, y_train_)
# lr = joblib.load('./Sentiment Classification/outdata/test.model')
joblib.dump(lr, './Sentiment Classification/outdata/test.model')
y_pred = lr.predict(X_test)
precision, recall, fscore, support = score(y_test, y_pred)
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))
TP = 0
FP = 0
FN = 0
TN = 0
p = 0
for pred, true in zip(y_pred, y_test):
    if pred == p and true == p:
        TP += 1
    elif pred == p and true != p:
        FP += 1
    elif pred != p and true == p:
        FN += 1
    elif pred != p and true != p:
        TN += 1
print('TP = {}, FP = {}, FN = {}, FP = {}'.format(TP, FP, FN, TN))