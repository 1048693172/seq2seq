import torch
from torch.autograd import Variable


class Predictor(object):

    def __init__(self, model, bl_vocab, label1_vocab, label2_vocab, label3_vocab):
        """
        Predictor class to evaluate for a given model.
        Args:
            model (seq2seq.models): trained model. This can be loaded from a checkpoint
                using `seq2seq.util.checkpoint.load`
            src_vocab (seq2seq.dataset.vocabulary.Vocabulary): source sequence vocabulary
            tgt_vocab (seq2seq.dataset.vocabulary.Vocabulary): target sequence vocabulary
        """
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model.cpu()
        self.model.eval()
        self.bl_vocab = bl_vocab
        # self.jz_vocab = jz_vocab
        self.label1_vocab = label1_vocab
        self.label2_vocab = label2_vocab
        self.label3_vocab = label3_vocab

    def get_decoder_features(self, src_seq_bl):
        src_id_seq_bl = torch.LongTensor([self.bl_vocab.stoi[tok] for tok in src_seq_bl]).view(1, -1)
        # src_id_seq_jz = torch.LongTensor([self.jz_vocab.stoi[tok] for tok in src_seq_jz]).view(1, -1)
        if torch.cuda.is_available():
            src_id_seq_bl = src_id_seq_bl.cuda()
            # src_id_seq_jz = src_id_seq_jz.cuda()

        with torch.no_grad():
            # softmax_list, _, other = self.model(src_id_seq_bl, src_id_seq_jz, [len(src_seq_bl)], [len(src_seq_jz)])
            other1, other2, other3 = self.model(src_id_seq_bl, [len(src_seq_bl)])
            softmax_list1, _, other1 = other1
            softmax_list2, _, other2 = other2
            softmax_list3, _, other3 = other3
        return other1, other2, other3

    def predict(self, src_seq_bl):
        """ Make prediction given `src_seq` as input.

        Args:
            src_seq (list): list of tokens in source language

        Returns:
            tgt_seq (list): list of tokens in target language as predicted
            by the pre-trained model
        """
        other1, other2, other3 = self.get_decoder_features(src_seq_bl)

        length1 = other1['length'][0]
        length2 = other2['length'][0]
        length3 = other3['length'][0]

        tgt_id_seq1 = [other1['sequence'][di][0].data[0] for di in range(length1)]
        tgt_seq1 = [self.label1_vocab.itos[tok] for tok in tgt_id_seq1]
        tgt_id_seq2 = [other2['sequence'][di][0].data[0] for di in range(length2)]
        tgt_seq2 = [self.label2_vocab.itos[tok] for tok in tgt_id_seq2]
        tgt_id_seq3 = [other3['sequence'][di][0].data[0] for di in range(length3)]
        tgt_seq3 = [self.label3_vocab.itos[tok] for tok in tgt_id_seq3]

        return tgt_seq1, tgt_seq2, tgt_seq3

    def predict_n(self, src_seq_bl,src_seq_jz, n=1):
        """ Make 'n' predictions given `src_seq` as input.

        Args:
            src_seq (list): list of tokens in source language
            n (int): number of predicted seqs to return. If None,
                     it will return just one seq.

        Returns:
            tgt_seq (list): list of tokens in target language as predicted
                            by the pre-trained model
        """
        other = self.get_decoder_features(src_seq_bl)

        result = []
        for x in range(0, int(n)):
            length = other['topk_length'][0][x]
            tgt_id_seq = [other['topk_sequence'][di][0, x, 0].data[0] for di in range(length)]
            tgt_seq = [self.tgt_vocab.itos[tok] for tok in tgt_id_seq]
            result.append(tgt_seq)

        return result
