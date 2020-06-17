# -*- coding:utf-8 -*-
from collections import defaultdict
from gensim import corpora
from gensim import models
#语料库
documents = [
    "Arthur Pendragon was the greatest king in the history of the Great Britain",
    "He is known as the Once and Future King or King Arthur",
    "There was probably a real Arthur though historians know little about him",
    "Who led the British people in battles against the Anglo-Saxons",
    "The real Arthur probably won some victories over the Anglo-Saxons",
    "But he was not nearly as great as the King Arthur",
    "Arthur was the leader of the Knights of the Round Table",
    "which was the place aggregated Lancelot and Gawain and Mordred and other knights",
    "Although he brought freedom and glory to the Great Britain",
    "He was rebelled by Lancelot and also Mordred",
    "Finally he died because of Mordred's rebellion",
    "With the respect and commemoration of Arthur and his knights",
    "Lots of people began to circulate his story",
    "These people passed on the earliest tales about Arthur by word of mouth",
    "Giving audience a recall of knights in the Middle Ages and a culture of chivalry",
    "Many people believes Arthur went to Avalon after his death",
    "Which stands for a forever ideal land",
    "But Avalon is just a story",
    "The Brirish bulit a tomb for him at Glastonbury Abbey",
    "Which means the forever king Arthur rests in peace here",
]
stopf = open('D:/Python/recommend/stoplist.txt','r')   #打开停止词文档，在分词时需跳过这些中停止词
stoptxt = stopf.read()
stopset = set(stoptxt.split())
word_list = [
    [word for word in document.lower().split() if word not in stopset]
    for document in documents
]

dictionary = corpora.Dictionary(word_list)     #生成字典，为语料库中每个不重复的词语分配一个id
corpus = [dictionary.doc2bow(text) for text in word_list]
print(dictionary.token2id)           # 打印所有单词的id
tfidf = models.TfidfModel(corpus)   #训练tf-idf模型
tfidf_vec = []                      

for i in range(len(documents)):          # 训练完成后得到各词语的tfidf值
    string = documents[i]
    string_bow = dictionary.doc2bow(string.lower().split())
    string_tfidf = tfidf[string_bow]
    tfidf_vec.append(string_tfidf)

for tfdif  in tfidf_vec: print(tfdif)    #打印出不同文本所对应的tf-idf值
