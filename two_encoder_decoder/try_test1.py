# _*_coding:utf-8 _*_
import os
import argparse
import logging
import jieba
import torch
import random,json
from torch.autograd import Variable

from torch.optim.lr_scheduler import StepLR
import torchtext

import seq2seq
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.loss import Perplexity
from seq2seq.optim import Optimizer
from seq2seq.dataset import BLField, JZField,TargetField
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

# Sample usage:
#     # training
#     python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH
#     # resuming from the latest checkpoint of the experiment
#      python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --resume
#      # resuming from a specific checkpoint
#      python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --load_checkpoint $CHECKPOINT_DIR

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', action='store', dest='train_path',
                    help='Path to train data')
parser.add_argument('--dev_path', action='store', dest='dev_path',
                    help='Path to dev data')
parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./experiment1',
                    help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
                    help='The name of the checkpoint to load, usually an encoded time string')
parser.add_argument('--resume', action='store_true', dest='resume',
                    default=False,
                    help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--log-level', dest='log_level',
                    default='info',
                    help='Logging level.')

opt = parser.parse_args()

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)

if opt.load_checkpoint is not None:
    logging.info("loading checkpoint from {}".format(os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)))
    checkpoint_path = os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)
    checkpoint = Checkpoint.load(checkpoint_path)
    seq2seq = checkpoint.model
    input_vocab1 = checkpoint.input_vocab1
    output_vocab1 = checkpoint.output_vocab1
    input_vocab2 = checkpoint.input_vocab2
    output_vocab2 = checkpoint.output_vocab2
else:
    # Prepare dataset
    bl = BLField()
    jz = JZField()
    label = TargetField()
    label2 = TargetField()
    max_len = 6000

    def len_filter(example):
        return len(example.bl) <= max_len and len(example.jz) <= max_len and len(example.label) \
               <= max_len and len(example.label2) <= max_len

    train = torchtext.data.TabularDataset(
        # path=opt.train_path
        # path='./data/toy_reverse/train/rw1cail.txt'
        path='source.json', format='json',
        fields={'bl': ('bl', bl),
                'jz': ('jz', jz),
                'label': ('label', label),
                'label2': ('label2', label2)},
        filter_pred=len_filter
    )

    dev = torchtext.data.TabularDataset(
        # path=opt.dev_path
        # path='./data/toy_reverse/dev/rw1dev_wb1.txt'
        path='source.json', format='json',
        fields={'bl': ('bl', bl),
                'jz': ('jz', jz),
                'label': ('label', label),
                'label2': ('label2', label2)},
        filter_pred=len_filter
    )
    test = torchtext.data.TabularDataset(
        # path=opt.dev_path
        # path='./data/toy_reverse/dev/rw1dev_wb1.txt'
        path='source.json', format='json',
        fields={'bl': ('bl', bl),
                'jz': ('jz', jz),
                'label': ('label', label),
                'label2': ('label2', label2)},
        filter_pred=len_filter
    )

    bl.build_vocab(train, max_size=500000)
    jz.build_vocab(train, max_size=50000)
    label.build_vocab(train, max_size=5000)
    label2.build_vocab(train, max_size=5000)
    bl_vocab = bl.vocab
    jz_vocab = jz.vocab
    label1_vocab = label.vocab
    label2_vocab = label2.vocab


    # NOTE: If the source field name and the target field name
    # are different from 'src' and 'tgt' respectively, they have
    # to be set explicitly before any training or inference
    # seq2seq.src_field_name = 'src'
    # seq2seq.tgt_field_name = 'tgt'

    # Prepare loss
    weight1 = torch.ones(len(label.vocab))
    pad1 = label.vocab.stoi[label.pad_token]
    loss1 = Perplexity(weight1, pad1)

    weight2 = torch.ones(len(label2.vocab))
    pad2 = label2.vocab.stoi[label2.pad_token]
    loss2 = Perplexity(weight2, pad2)
    if torch.cuda.is_available():
        loss1.cuda()
        loss2.cuda()



    seq2seq = None
    optimizer = None
    if not opt.resume:
        # Initialize model
        hidden_size = 256
        bidirectional = True

        encoder1 = EncoderRNN(len(bl.vocab), max_len, hidden_size,
                              bidirectional=bidirectional, variable_lengths=True)
        encoder2 = EncoderRNN(len(jz.vocab), max_len, hidden_size,
                              bidirectional=bidirectional, variable_lengths=False)

        decoder1 = DecoderRNN(len(label.vocab), max_len, hidden_size * 2 if bidirectional else hidden_size,
                              dropout_p=0.5, use_attention=True, bidirectional=bidirectional,
                              eos_id=label.eos_id, sos_id=label.sos_id)

        decoder2 = DecoderRNN(len(label2.vocab), max_len, hidden_size * 2 if bidirectional else hidden_size,
                              dropout_p=0.5, use_attention=True, bidirectional=bidirectional,
                              eos_id=label2.eos_id, sos_id=label2.sos_id)

        seq2seq = Seq2seq(encoder1, encoder2, decoder1, decoder2)
        if torch.cuda.is_available():
            seq2seq.cuda()

        for param in seq2seq.parameters():
            param.data.uniform_(-0.08, 0.08)

        # Optimizer and learning rate scheduler can be customized by
        # explicitly constructing the objects and pass to the trainer.
        #
        # optimizer = Optimizer(torch.optim.Adam(seq2seq.parameters()), max_grad_norm=5)
        # scheduler = StepLR(optimizer.optimizer, 1)
        # optimizer.set_scheduler(scheduler)

    # train
    t = SupervisedTrainer(loss1, loss2, batch_size=16,
                          checkpoint_every=2000,
                          print_every=1000, expt_dir=opt.expt_dir)

    seq2seq = t.train(seq2seq, train, dev,
                      num_epochs=6,
                      optimizer=optimizer,
                      teacher_forcing_ratio=0,
                      resume=opt.resume)
print('done')
torch.save(seq2seq, "./testjson.model")
torch.save(bl_vocab, "./testbl.input_vocab")
torch.save(jz_vocab, "./testjz.input_vocab")
torch.save(label1_vocab, "./testlabel.input_vocab")
torch.save(label2_vocab, "./testlabel.input_vocab")
torch.save(output_vocab1, "./test.output_vocab1")
torch.save(output_vocab2, "./test.output_vocab2")
# model = torch.load("./testjson.model")
# bl_vocab = torch.load("./testbl.input_vocab")
# jz_vocab = torch.load("./testjz.input_vocab")
# label_vocab = torch.load("./testlabel.input_vocab")
#
# predictor = Predictor(model, bl_vocab, jz_vocab, label_vocab)
#
#
#
#
# #
# import sys
#
# origin = sys.stdout
#
# frr = open('../syjieguo.txt', 'a',encoding='utf-8')
#
# sys.stdout = frr
# labels_right = []
# labels_predictor = []
# texts_bl = []
# texts_jz = []
# # with open("E:/公安项目/第二期/警综/场所/jzyp_test.txt",'r',encoding='utf-8') as fr:
# with open("shiyan.json", 'r', encoding='utf-8') as fr:
#     for line in fr:
#         line = json.loads(line)
#         texts_bl.append(line['bl'])
#         texts_jz.append(line['jz'])
#         labels_right.append(line['label'])
#         # print(line['label'])
#
#
# for seq_str1,seq_str2 in zip(texts_bl,texts_jz):
#         # seq_str = raw_input("Type in a source sequence:")
#     seq1 = seq_str1.strip().split()
#     seq2 = seq_str2.strip().split()
#     predict_label = predictor.predict(seq1,seq2)
#     labels_predictor.append(" ".join(str(i) for i in set(predict_label[:-1])))
#
#
# a = labels_predictor
# bel = list(set(a))
# print(bel)
# print(len(a))
#
#
#
# text_labels = list(set(labels_right))
# A = dict.fromkeys(text_labels,0)  #预测正确的各个类的数目
# B = dict.fromkeys(text_labels,0)   #测试数据集中各个类的数目
# C = dict.fromkeys(bel,0) #预测结果中各个类的数目
# for i in range(0,len(labels_right)):
#     B[labels_right[i]] += 1
#     C[a[i]] += 1
#     if labels_right[i] == a[i]:
#         A[labels_right[i]] += 1
#
# print("预测正确的各个类的数目:");print(A)
# print("测试数据集中各个类的数目:");print(B)
# print("预测结果中各个类的数目:");print(C)
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
#
#
# print(sum(A.values())/float(len(a)))
#
# sys.stdout = origin
# frr.close()

