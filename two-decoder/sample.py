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
from seq2seq.dataset import BLField, JZField, TargetField
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
    label = TargetField()
    label2 = TargetField()
    max_len = 6000

    def len_filter(example):
        return len(example.bl) <= max_len and len(example.label) \
               <= max_len and len(example.label2) <= max_len

    train = torchtext.data.TabularDataset(
        # path=opt.train_path
        # path='./data/toy_reverse/train/rw1cail.txt'
        path='./data/source_train.json', format='json',
        fields={'bl': ('bl', bl),
                'label': ('label', label),
                'label2': ('label2', label2)},
        filter_pred=len_filter
    )

    dev = torchtext.data.TabularDataset(
        # path=opt.dev_path
        # path='./data/toy_reverse/dev/rw1dev_wb1.txt'
        path='./data/source_dev.json', format='json',
        fields={'bl': ('bl', bl),
                'label': ('label', label),
                'label2': ('label2', label2)},
        filter_pred=len_filter
    )
    test = torchtext.data.TabularDataset(
        # path=opt.dev_path
        # path='./data/toy_reverse/dev/rw1dev_wb1.txt'
        path='./data/source_test.json', format='json',
        fields={'bl': ('bl', bl),
                'label': ('label', label),
                'label2': ('label2', label2)},
        filter_pred=len_filter
    )

    bl.build_vocab(train, max_size=500000)
    label.build_vocab(train, max_size=5000)
    label2.build_vocab(train, max_size=5000)
    bl_vocab = bl.vocab
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

        decoder1 = DecoderRNN(len(label.vocab), max_len, hidden_size * 2 if bidirectional else hidden_size,
                              dropout_p=0.5, use_attention=True, bidirectional=bidirectional,
                              eos_id=label.eos_id, sos_id=label.sos_id)

        decoder2 = DecoderRNN(len(label2.vocab), max_len, hidden_size * 2 if bidirectional else hidden_size,
                              dropout_p=0.5, use_attention=True, bidirectional=bidirectional,
                              eos_id=label2.eos_id, sos_id=label2.sos_id)

        seq2seq = Seq2seq(encoder1, decoder1, decoder2)
        if torch.cuda.is_available():
            seq2seq.cuda()

        for param in seq2seq.parameters():
            param.data.uniform_(-0.08, 0.08)
    # train
    t = SupervisedTrainer(loss1, loss2, batch_size=16,
                          checkpoint_every=2000,
                          print_every=100, expt_dir=opt.expt_dir)

    seq2seq = t.train(seq2seq, train, dev,
                      num_epochs=4,
                      optimizer=optimizer,
                      teacher_forcing_ratio=0,
                      resume=opt.resume)
    print('done')
torch.save(seq2seq, "./testjson.model")
torch.save(bl_vocab, "./testbl.input_vocab")
torch.save(label1_vocab, "./testlabel1.input_vocab")
torch.save(label2_vocab, "./testlabel2.input_vocab")
predictor = Predictor(seq2seq, bl_vocab, label1_vocab, label2_vocab)
while True:
    seq_str1 = input("Type in a source bl sequence:")
    seq1 = seq_str1.strip().split()
    print(predictor.predict(seq1))
