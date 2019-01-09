# _*_coding:utf-8 _*_
import jieba
import torch
import random
from torch.autograd import Variable

class Batch_Predictor(object):
    def __init__(self, model1, src_vocab, tgt_vocab, tgt_vocab2, tgt_vocab3):
        if torch.cuda.is_available():
            self.model1 = model1.cuda()
        else:
            self.model1 = model1.cpu()
        self.model1.eval()
        self.src_vocab1 = src_vocab
        self.tgt_vocab1 = tgt_vocab
        self.tgt_vocab2 = tgt_vocab2
        self.tgt_vocab3 = tgt_vocab3
        self.batch_size = 32

    def sortlength(self, L):  # L为列表
        n = len(L)
        for i in range(n):
            k = i
            j = i + 1
            while j < n:
                if len(L[k]) > len(L[j]):
                    k = j
                j = j + 1
            if i != k:
                temp = L[k]
                L[k] = L[i]
                L[i] = temp

    def splitlength(self, L):
        a1 = []
        A = {}
        for l in L:
            if len(l) not in A.keys():
                A[len(l)] = [l]
            else:
                A[len(l)] += [l]
        for a in A.values():
            if len(a) <= self.batch_size:
                a1 += [a]
            else:
                a1 += [a[i:i + self.batch_size] for i in range(0, len(a), self.batch_size)]
        return a1

    def predict_no_cut_textlist(self, content):
        result = []
        x1 = []
        i = 0
        while i < len(content):
            cut_word = jieba.cut(str(content[i]), cut_all=True, HMM=False)
            seg = ' '.join(cut_word).strip().split()
            x1.append([self.src_vocab1.stoi[tok] for tok in seg] + [i])
            i += 1

        self.sortlength(x1)
        tempo1 = self.splitlength(x1)
        B = {}
        j = 0
        while j < len(tempo1):
            x1 = tempo1[j]
            a1 = torch.LongTensor(x1[0][:-1]).view(1, -1)
            y = [len(x1[0][:-1])]
            xlocal = [x1[0][-1]]
            i = 1
            while i < len(x1):
                a1 = torch.cat((a1, torch.LongTensor(x1[i][:-1]).view(1, -1)), 0)
                y.append(len(x1[i][:-1]))
                xlocal.append(x1[i][-1])
                i += 1
            if torch.cuda.is_available():
                a1 = a1.cuda()
            with torch.no_grad():
                doutputs1, dhidden1, other1 = self.model1(Variable(a1, volatile=True), y)

            length1 = other1['length']
            seqlist1 = other1['sequence']

            i = 0
            A = {}
            while i < len(xlocal):
                tgt_id_seq1 = [seqlist1[di][i].data[0] for di in range(length1[i])]
                tgt_seq1 = [self.tgt_vocab1.itos[tok] for tok in tgt_id_seq1]

                A[xlocal[i]] =  tgt_seq1[:-1]
                i += 1
            j += 1
            B.update(A)
        c = sorted(B.keys())
        for key in c:
            result.append(B[key])
        return result

    def predict_cutseq(self, content):
        result = []
        x1 = []
        i=0
        x1.append([self.src_vocab1.stoi[tok] for tok in content] + [i])

        self.sortlength(x1)
        tempo1 = self.splitlength(x1)
        B = {}
        j = 0
        while j < len(tempo1):
            x1 = tempo1[j]
            a1 = torch.LongTensor(x1[0][:-1]).view(1, -1)
            y = [len(x1[0][:-1])]
            xlocal = [x1[0][-1]]
            i = 1
            while i < len(x1):
                a1 = torch.cat((a1, torch.LongTensor(x1[i][:-1]).view(1, -1)), 0)
                y.append(len(x1[i][:-1]))
                xlocal.append(x1[i][-1])
                i += 1
            if torch.cuda.is_available():
                a1 = a1.cuda()
            with torch.no_grad():
                doutputs1, dhidden1, other1 = self.model1(Variable(a1, volatile=True), y)
            length1 = other1['length']
            seqlist1 = other1['sequence']

            i = 0
            A = {}
            while i < len(xlocal):
                tgt_id_seq1 = [seqlist1[di][i].data[0] for di in range(length1[i])]
                tgt_seq1 = [self.tgt_vocab1.itos[tok] for tok in tgt_id_seq1]

                A[xlocal[i]] = tgt_seq1[:-1]
                i += 1
            j += 1
            B.update(A)
        c = sorted(B.keys())
        for key in c:
            result.append(B[key])
        return result

    def predict_cut_textlist(self, content):
        result = []
        x1 = []
        i = 0
        while i < len(content):
            aa = content[i].split()
            x1.append([self.src_vocab1.stoi[tok] for tok in aa] + [i])
            i += 1

        self.sortlength(x1)
        tempo1 = self.splitlength(x1)
        B = {}
        j = 0
        while j < len(tempo1):
            x1 = tempo1[j]
            a1 = torch.LongTensor(x1[0][:-1]).view(1, -1)
            y = [len(x1[0][:-1])]
            xlocal = [x1[0][-1]]
            i = 1
            while i < len(x1):
                a1 = torch.cat((a1, torch.LongTensor(x1[i][:-1]).view(1, -1)), 0)
                y.append(len(x1[i][:-1]))
                xlocal.append(x1[i][-1])
                i += 1
            if torch.cuda.is_available():
                a1 = a1.cuda()
            with torch.no_grad():
                other1, other2, other3 = self.model1(Variable(a1, volatile=True), y)
            length1 = other1[2]['length']
            seqlist1 = other1[2]['sequence']
            length2 = other2[2]['length']
            seqlist2 = other2[2]['sequence']
            length3 = other3[2]['length']
            seqlist3 = other3[2]['sequence']

            i = 0
            A = {}
            while i < len(xlocal):
                tgt_id_seq1 = [seqlist1[di][i].data[0] for di in range(length1[i])]
                tgt_seq1 = [self.tgt_vocab1.itos[tok] for tok in tgt_id_seq1]
                tgt_id_seq2 = [seqlist2[di][i].data[0] for di in range(length2[i])]
                tgt_seq2 = [self.tgt_vocab2.itos[tok] for tok in tgt_id_seq2]
                tgt_id_seq3 = [seqlist3[di][i].data[0] for di in range(length3[i])]
                tgt_seq3 = [self.tgt_vocab3.itos[tok] for tok in tgt_id_seq3]

                A[xlocal[i]] = [tgt_seq1[:-1], tgt_seq2[:-1], tgt_seq3[:-1]]
                i += 1
            j += 1
            B.update(A)
        c = sorted(B.keys())
        for key in c:
            result.append(B[key])
        return result
