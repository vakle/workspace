# -*- coding:utf-8 -*-
import os
from gensim import corpora
from gensim import models
from gensim import similarities
stopf = open('D:/Python/recommend/stoplist.txt','r')   #打开停止词文档，在分词时需跳过这些中停止词
stoptxt = stopf.read()
stopset = set(stoptxt.split())

train_set = []     #训练集
walk = os.walk('D:/Python/recommend/test/text/')
for root, dirs, files in walk:
   for name in files:              #遍历所有文档
        fpath = os.path.join(root, name)
        f = open(fpath, 'r',encoding='utf-8',errors='ignore')
        raw = f.read().lower()      #依次读取并转换为小写
        word_list = []
        for word in raw.split(' '):
          if word not in stopset:
            word_list.append(word)      #去除stoplist的中止词后将其余词语加入wordlist
        train_set.append(word_list)     #将每一个文档对应的wordlist加入训练集

dictionary = corpora.Dictionary(train_set)  #生成字典，为语料库中每个不重复的词语分配一个id
corpus = [dictionary.doc2bow(text) for text in train_set]   
tfidf = models.TfidfModel(corpus)      #训练出tf-idf模型，tf-idf模型为lda模型基础
corpus_tfidf = tfidf[corpus]           
lda = models.LdaModel(corpus_tfidf, id2word = dictionary, num_topics = 12)   #构建lda模型，主题数为12个
corpus_lda = lda[corpus_tfidf]       

for i in range(0, 12):    #打印主题
	print (lda.print_topic(i))
