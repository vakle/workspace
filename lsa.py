# -*- coding:utf-8 -*-
from collections import defaultdict
from gensim import corpora
from gensim import models
from gensim import similarities
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
stopf = open('D:/Python/recommend/stoplist.txt','r')   #打开停止词文档，在分词时需删除这些停止词
stoptxt = stopf.read()
stopset = set(stoptxt.split())
texts = [
    [word for word in document.lower().split() if word not in stopset]
    for document in documents
]
frequency = defaultdict(int)   #去除那些仅出现一次的词语，保留剩下的
for text in texts:
    for token in text:
        frequency[token] += 1
texts = [
    [token for token in text if frequency[token] > 1]
    for text in texts
]

dictionary = corpora.Dictionary(texts)   #生成字典，为语料库中每个不重复的词语分配一个id
corpus = [dictionary.doc2bow(text) for text in texts]   

lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)  #训练lsa模型
doc = "Who is the most famous king in the world"                   #新文本，用于推荐结果测试
vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lsi = lsi[vec_bow]  # 转换为LSI空间

index = similarities.MatrixSimilarity(lsi[corpus])  #将原始语料转换为LSI空间并添加索引
sims = index[vec_lsi]  #相似度查询

sims = sorted(enumerate(sims), key=lambda item: -item[1])   #将相似度查询结果排序，排名越靠前的相似度越高
for i, s in enumerate(sims):            #打印所以文本的索引以及相似度
    print(s, documents[i])