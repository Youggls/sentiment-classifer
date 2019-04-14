from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from gensim.models import word2vec
from sklearn import metrics
from sklearn.externals import joblib
import numpy

class SentimentClassifier:
    __isInitial = False
    __isTrain = False
    __test_size = 0.25
    __feature = 100
    __vec_model = word2vec.Word2Vec()
    __classifer = object()
    __X_test = object()
    __y_test = object()

    def __init__(self):
        self.__isInitial = False
        self.__isTrain = False

    def train(self, lines, test_size=0.25, feature=100):
        if self.__isInitial == False:
            raise RuntimeError("The Classifier has not been initialized!")
        
        self.__feature = feature
        lines = numpy.array(lines)
        m, n = lines.shape
        X = list()
        label = list()
        for i in range(m):
            if lines[i][1] in self.__vec_model:
                X.append(self.__vec_model[lines[i][1]])
                label.append(int(lines[i][2]))

        X = numpy.array(X)
        label = numpy.array(label)
        X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=self.__test_size, random_state=33)

        self.__classifer = LogisticRegression()
        self.__classifer.fit(X_train, y_train)
        self.__X_test = X_test
        self.__y_test = y_test
        self.__isTrain = True
        
    def predict(self, word):
        if self.__isTrain == False:
            raise RuntimeError('The Classifer has not been trained!')
        elif self.__isInitial == False:
            raise RuntimeError('The Classifer has not been initialized!')
        
        if word not in self.__vec_model:
            return self.__classifer.predict_proba([self.__vec_model[word]])

    def __getitem__(self, word):
        if self.__isTrain == False:
            raise RuntimeError('The Classifer has not been trained!')
        elif self.__isInitial == False:
            raise RuntimeError('The Classifer has not been initialized!')
        
        if word not in self.__vec_model:
            print('The word is not in vocabulary!')
        
        return self.predict(word)

    def score(self):
        if self.__isTrain == False:
            raise RuntimeError('The Classifer has not been trained!')
        elif self.__isInitial == False:
            raise RuntimeError('The Classifer has not been initialized!')
        
        y_pred = self.__classifer.predict(self.__X_test)
        return precision_recall_fscore_support(self.__y_test, y_pred)

    def save(self, vec_path, model_path):
        self.save_vec(vec_path)
        self.save_model(model_path)

    def save_vec(self, vec_path):
        if self.__isInitial == False:
            raise RuntimeError('The Classifer has not been initialized!')
        self.__vec_model.save(vec_path)

    def save_model(self, model_path):
        if self.__isTrain == False:
            raise RuntimeError('The Classifer has not been trained!')
        joblib.dump(self.__classifer, model_path)
    
    def load_vec(self, vec_path):
        self.__vec_model = word2vec.Word2Vec.load(vec_path)
        self.__isInitial = True

    def load_model(self, model_path):
        self.__classifer = joblib.load(model_path)
        self.__isTrain = True

    def load(self, vec_path, model_path):
        load_vec(vec_path)
        load_model(model_path)