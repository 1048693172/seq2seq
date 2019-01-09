from seq2seq.evaluator import Predictor
import torch
import json
from seq2seq.util.checkpoint import Checkpoint

# checkpoint = Checkpoint.load("./experiment1/checkpoints/2018_12_05_11_06_14")
# model = checkpoint.model
# bl_vocab = checkpoint.input_vocab1
# jz_vocab = checkpoint.input_vocab2
# label1_vocab = checkpoint.output_vocab1
# label2_vocab = checkpoint.output_vocab2
# # torch.save(model, "testjson.model")
# # torch.save(input_vocab1, "testbl.input_vocab")
# # torch.save(input_vocab2, "testjz.input_vocab")
# # torch.save(output_vocab1, "testlabel1.input_vocab")
# # torch.save(output_vocab2, "testlabel2.input_vocab")
model = torch.load("./models/testjson.model")
bl_vocab = torch.load("./models/testbl.input_vocab")
jz_vocab = torch.load("./models/testjz.input_vocab")
label1_vocab = torch.load("./models/testlabel1.input_vocab")
label2_vocab = torch.load("./models/testlabel2.input_vocab")
predictor = Predictor(model, bl_vocab, jz_vocab, label1_vocab, label2_vocab)

# while True:
#     seq_str1 = input("Type in a source bl sequence:")
#     seq1 = seq_str1.strip().split()
#     seq_str2 = input("Type in a source jz sequence:")
#     seq2 = seq_str2.strip().split()
#     print(predictor.predict(seq1, seq2))

labels_right1 = []
labels_right2 = []
labels_predictor1 = []
labels_predictor2 = []
texts_bl = []
texts_jz = []

with open("./data/source_test.json", 'r', encoding='utf-8') as fr:
    for line in fr:
        line = json.loads(line)
        texts_bl.append(line['bl'])
        texts_jz.append(line['jz'])
        labels_right1.append(line['label'])
        labels_right2.append(line['label2'])

for seq_str1, seq_str2 in zip(texts_bl, texts_jz):
    seq1 = seq_str1.strip().split()
    seq2 = seq_str2.strip().split()
    predict_label1, predict_label2 = predictor.predict(seq1, seq2)
    labels_predictor1.append(" ".join(str(i) for i in set(predict_label1[:-1])))
    labels_predictor2.append(" ".join(str(i) for i in set(predict_label2[:-1])))
j = 0
k = 0
b = 0
with open("./data/result_source.txt", 'w') as f:
    for i in range(len(texts_bl)):
        if labels_right1[i] == labels_predictor1[i]:
            s = 'right1'
            j += 1
        if labels_right2[i] == labels_predictor2[i]:
            s = 'right2'
            k += 1
        if labels_right1[i] == labels_predictor1[i] and labels_right2[i] == labels_predictor2[i]:
            s = 'both-right'
            b += 1
        if labels_right1[i] != labels_predictor1[i] and labels_right2[i] != labels_predictor2[i]:
            s = 'false'
        # dd = labels_predictor1[i]
        try:
            f.write(labels_right1[i] + "\t" + labels_predictor1[i] + "\t" + "\t" + labels_right2[i] + "\t" +
                    labels_predictor2[i] + "\t" + s)
            f.write('\n')
        except:
            pass
    f.write(str(j/len(texts_bl)))
    f.write('\n')
    f.write(str(k/len(texts_bl)))
    f.write('\n')
    f.write(str(b/len(texts_bl)))

# a1 = labels_predictor1
# bel1 = list(set(a1))
# print(bel1)
# print(len(a1))
#
#
#
# text_labels = list(set(labels_right1))
# A = dict.fromkeys(text_labels, 0)  #预测正确的各个类的数目
# B = dict.fromkeys(text_labels, 0)   #测试数据集中各个类的数目
# C = dict.fromkeys(bel1, 0) #预测结果中各个类的数目
# for i in range(0, len(labels_right1)):
#     B[labels_right1[i]] += 1
#     C[a1[i]] += 1
#     if labels_right1[i] == a1[i]:
#         A[labels_right1[i]] += 1
#
# print("预测正确的各个类的数目:"); print(A)
# print("测试数据集中各个类的数目:"); print(B)
# print("预测结果中各个类的数目:"); print(C)
# #计算准确率，召回率，F值
# for key in B:
#     try:
#         r = float(A[key]) / float(B[key])
#         p = float(A[key]) / float(C[key])
#         f = p * r * 2 / (p + r)
#         # print(("%s:\t p:%f\t r:%f\t f:%f") % (key,p,r,f))
#         print(key, "right:", A.get(key, 0), "real:", B.get(key, 0), "predict:", C.get(key, 0), ("p: %f\t r: %f\t f: %f") % (p, r, f))
#     except:
#         print(("error:"+key), "right:", A.get(key, 0), "real:", B.get(key, 0), "predict:", C.get(key, 0))
#
# print(sum(A.values())/float(len(a1)))
