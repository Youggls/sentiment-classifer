from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from gensim.models import word2vec
from sklearn import metrics
import numpy

class SentimentClassifier:
    __isInitial = False
    __isTrain = False
    __test_size = 0.25
    __feature = 100
    __model = word2vec.Word2Vec()
    __classifier = object()
    __X_test = object()
    __y_test = object()

    def __init__(self):
        self.__isInital = False
        self.__isTrain = False

    def __init__(self, sentence, min=5, nthread=1):
        self.__model = word2vec.Word2Vec(sentence, min_count=min, workers=nthread)
        self.__isInitial = True
        self.__isTrain = False

    def __init__(self, sentence, lines, test_size=0.25, min=5, nthread=1, feature=100):
        self.__model = word2vec.Word2Vec(sentence, min_count=min, workers=nthread)
        self.__isInitial = True
        self.Train(lines, test_size=test_size, feature=feature)
        self.__isTrain = True
    
    def __init__(self, model_path, lines, test_size=0.25):
        self.__model = word2vec.Word2Vec.load(model_path)
        self.__isInitial = True
        feature = self.__model.wv.vector_size
        self.Train(lines, test_size=test_size, feature=feature)
        self.__isTrain = True

    def Train(self, lines, test_size=0.25, feature=100):
        if self.__isInitial == False:
            raise RuntimeError("The Classifier has not been initialized!")
        
        self.__feature = feature
        lines = numpy.array(lines)
        m, n = lines.shape
        X = list()
        label = list()
        for i in range(m):
            if lines[i][1] in self.__model:
                X.append(self.__model[lines[i][1]])
                label.append(int(lines[i][2]))

        X = numpy.array(X)
        label = numpy.array(label)
        X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=self.__test_size, random_state=33)

        self.__classifer = LogisticRegression()
        self.__classifer.fit(X_train, y_train)
        self.__X_test = X_test
        self.__y_test = y_test

    def predict(self, word):
        if word not in self.__model:
            return self.__classifer.predict_proba([self.__model[word]])

    def __getitem__(self, word):
        if word not in self.__model:
            print('The word is not in vocabulary!')
        
        return self.predict(word)

    def score(self):
        if self.__isTrain == False:
            raise RuntimeError('The Classifer has not been trained!')
        y_pred = self.__classifer.predict(self.__X_test)
        return precision_recall_fscore_support(self.__y_test, y_pred)

    def save(self):
        if self.__isInitial == Flase:
            raise RuntimeError('The Classifer has not been initialized!')
        
        self.__model.save()
    
    def load(self, model_path):
        self.__model = word2vec.Word2Vec.load(model_path)