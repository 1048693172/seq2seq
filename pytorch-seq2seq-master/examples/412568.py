from gensim.models import word2vec

model = word2vec.Word2Vec.load('F:/word2vecWeixin/word2vec_wx')
file=open("F:\\wangkai\\pytorch-seq2seq-master\\data\\toy_reverse\\train\\16_train.txt",'r',encoding='utf-8')
aa=set()
for line in file.readlines():
    aa.add(a for a in line.split('\t')[0].split())

tt=0
for i in aa:
    try:
        vector = model.wv[i]
    except:
        tt+=1
print(len(aa))
print(tt)