from __future__ import print_function, division

import torch
import torchtext

import seq2seq
from seq2seq.loss import NLLLoss

class Evaluator(object):
    """ Class to evaluate models with given datasets.

    Args:
        loss (seq2seq.loss, optional): loss for evaluator (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for evaluator (default: 64)
    """

    def __init__(self, loss1=NLLLoss(), loss2=NLLLoss(), loss3=NLLLoss(), batch_size=64):
        self.loss1 = loss1
        self.loss2 = loss2
        self.loss3 = loss3
        self.batch_size = batch_size

    def evaluate(self, model, data1):
        """ Evaluate a model on given dataset and return performance.

        Args:
            model (seq2seq.models): model to evaluate
            data (seq2seq.dataset.dataset.Dataset): dataset to evaluate against

        Returns:
            loss (float): loss of the given model on the given dataset
        """
        model.eval()

        loss1 = self.loss1
        loss2 = self.loss2
        loss3 = self.loss3
        loss1.reset()
        loss2.reset()
        loss3.reset()
        match1 = 0
        total1 = 0
        match2 = 0
        total2 = 0
        match3 = 0
        total3 = 0

        device = None if torch.cuda.is_available() else -1
        batch_iterator1 = torchtext.data.BucketIterator(
            dataset=data1, batch_size=self.batch_size,
            sort=True, sort_key=lambda x: len(x.bl),
            device=device, train=False)

        tgt_vocab1 = data1.fields[seq2seq.label_field_name1].vocab
        pad1 = tgt_vocab1.stoi[data1.fields[seq2seq.label_field_name1].pad_token]
        tgt_vocab2 = data1.fields[seq2seq.label_field_name2].vocab
        pad2 = tgt_vocab2.stoi[data1.fields[seq2seq.label_field_name2].pad_token]
        tgt_vocab3 = data1.fields[seq2seq.label_field_name3].vocab
        pad3 = tgt_vocab3.stoi[data1.fields[seq2seq.label_field_name3].pad_token]

        # with torch.no_grad():
        for batch1 in batch_iterator1:
            input_variables1, input_lengths1 = getattr(batch1, seq2seq.bl_field_name)
            # input_variables2, input_lengths2 = getattr(batch1, seq2seq.jz_field_name)
            # input_variables3, input_lengths3 = getattr(batch3, seq2seq.src_field_name)
            target_variable1 = getattr(batch1, seq2seq.label_field_name1)
            target_variable2 = getattr(batch1, seq2seq.label_field_name2)
            target_variable3 = getattr(batch1, seq2seq.label_field_name3)
            if torch.cuda.is_available():
                input_variables1 = input_variables1.cuda()
                target_variable1 = target_variable1.cuda()
                target_variable2 = target_variable2.cuda()
                target_variable3 = target_variable3.cuda()
            d1, d2, d3 = model(input_variables1, input_lengths1.tolist(),
                               target_variable1, target_variable2, target_variable3)
            decoder_outputs1, decoder_hidden1, other1 = d1
            decoder_outputs2, decoder_hidden2, other2 = d2
            decoder_outputs3, decoder_hidden3, other3 = d3
            # Evaluation
            seqlist1 = other1['sequence']
            seqlist2 = other2['sequence']
            seqlist3 = other3['sequence']
            for step, step_output in enumerate(decoder_outputs1):
                target1 = target_variable1[:, step + 1]
                loss1.eval_batch(step_output.view(target_variable1.size(0), -1), target1)
                non_padding1 = target1.ne(pad1)
                correct1 = seqlist1[step].view(-1).eq(target1).masked_select(non_padding1).sum().item()
                match1 += correct1
                total1 += non_padding1.sum().item()

            for step, step_output in enumerate(decoder_outputs2):
                target2 = target_variable2[:, step + 1]
                loss2.eval_batch(step_output.view(target_variable2.size(0), -1), target2)
                non_padding2 = target2.ne(pad2)
                correct2 = seqlist2[step].view(-1).eq(target2).masked_select(non_padding2).sum().item()
                match2 += correct2
                total2 += non_padding2.sum().item()

            for step, step_output in enumerate(decoder_outputs3):
                target3 = target_variable3[:, step + 1]
                loss3.eval_batch(step_output.view(target_variable3.size(0), -1), target3)
                non_padding3 = target3.ne(pad3)
                correct3 = seqlist3[step].view(-1).eq(target3).masked_select(non_padding3).sum().item()
                match3 += correct3
                total3 += non_padding3.sum().item()

        if total1 == 0:
            accuracy1 = float('nan')
        else:
            accuracy1 = match1 / total1
        if total2 == 0:
            accuracy2 = float('nan')
        else:
            accuracy2 = match2 / total2
        if total3 == 0:
            accuracy3 = float('nan')
        else:
            accuracy3 = match3 / total3
        # return accuracy
        return loss1.get_loss(), loss2.get_loss(), loss3.get_loss(), accuracy1, accuracy2, accuracy3
