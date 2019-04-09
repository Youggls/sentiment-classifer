import jieba

with open('./Sentiment Classification/tianlong.txt', errors='ignore', encoding='utf-8') as fp:
   lines = fp.readlines()
   for line in lines:
       seg_list = jieba.cut(line)
       with open('./Sentiment Classification/tianlongout.txt', 'a', encoding='utf-8') as ff:
           for i in seg_list:
               ff.write(i)
               ff.write('\n')

from gensim.models import word2vec

# 加载语料
sentences = word2vec.Text8Corpus('./Sentiment Classification/tianlongout.txt')

# 训练模型
model = word2vec.Word2Vec(sentences)
model.save('./Sentiment Classification/tianlong.model')
# 选出最相似的10个词
for e in model.most_similar(positive=['乔峰'], topn=100):
   print(e[0], e[1])