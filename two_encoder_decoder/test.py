from seq2seq.evaluator import Predictor
import torch

model = torch.load("./testjson.model")
bl_vocab = torch.load("./testbl.input_vocab")
jz_vocab = torch.load("./testjz.input_vocab")
label1_vocab = torch.load("./testlabel1.input_vocab")
label2_vocab = torch.load("./testlabel2.input_vocab")

predictor = Predictor(model, bl_vocab, jz_vocab, label1_vocab, label2_vocab)
while True:
    seq_str1 = input("Type in a source bl sequence:")
    seq1 = seq_str1.strip().split()
    seq_str2 = input("Type in a source jz sequence:")
    seq2 = seq_str2.strip().split()
    print(predictor.predict(seq1, seq2))
