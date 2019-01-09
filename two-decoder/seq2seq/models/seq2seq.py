import torch.nn as nn
import torch.nn.functional as F
import torch

class Seq2seq(nn.Module):
    """ Standard sequence-to-sequence architecture with configurable encoder
    and decoder.

    Args:
        encoder (EncoderRNN): object of EncoderRNN
        decoder (DecoderRNN): object of DecoderRNN
        decode_function (func, optional): function to generate symbols from output hidden states (default: F.log_softmax)

    Inputs: input_variable, input_lengths, target_variable, teacher_forcing_ratio
        - **input_variable** (list, option): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the encoder.
        - **input_lengths** (list of int, optional): A list that contains the lengths of sequences
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)
        - **target_variable** (list, optional): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the decoder.
        - **teacher_forcing_ratio** (int, optional): The probability that teacher forcing will be used. A random number
          is drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0)

    Outputs: decoder_outputs, decoder_hidden, ret_dict
        - **decoder_outputs** (batch): batch-length list of tensors with size (max_length, hidden_size) containing the
          outputs of the decoder.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
          representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
          predicted token IDs, *KEY_INPUT* : target outputs if provided for decoding, *KEY_ATTN_SCORE* : list of
          sequences, where each list is of attention weights }.

    """

    def __init__(self, encoder1, decoder1, decoder2, decode_function=F.log_softmax):
        super(Seq2seq, self).__init__()
        self.encoder1 = encoder1
        # self.encoder2 = encoder2
        # self.encoder3 = encoder3
        self.decoder1 = decoder1
        self.decoder2 = decoder2
        self.decode_function = decode_function

    def flatten_parameters(self):
        self.encoder1.rnn.flatten_parameters()
        # self.encoder2.rnn.flatten_parameters()
        self.decoder1.rnn.flatten_parameters()
        self.decoder2.rnn.flatten_parameters()

    def forward(self, input_variable1,  input_lengths1=None,
                target_variable1=None, target_variable2=None, teacher_forcing_ratio=0):

        encoder_outputs1, encoder_hidden1 = self.encoder1(input_variable1, input_lengths1)
        # encoder_outputs2, encoder_hidden2 = self.encoder2(input_variable2, input_lengths2)
        # encoder_outputs3, encoder_hidden3 = self.encoder3(input_variable3, input_lengths3)
        # print('encoder_hidden1.size',encoder_hidden1.size())
        # print('encoder_hidden2.size',encoder_hidden2.size())
        # print('encoder_hidden3.size', encoder_hidden3.size())
        # print('encoder_outputs1.size', encoder_outputs1.size())
        # print('encoder_outputs2.size', encoder_outputs2.size())
        # print('encoder_outputs3.size', encoder_outputs3.size())
        # encoder_hidden = torch.cat((encoder_hidden1, encoder_hidden2, encoder_hidden3),1)
        # encoder_outputs = torch.cat((encoder_outputs1, encoder_outputs2, encoder_outputs3), 1)
        # encoder_hidden = torch.cat((encoder_hidden1, encoder_hidden2), 1)
        # encoder_outputs = torch.cat((encoder_outputs1, encoder_outputs2), 1)

        # print(encoder_hidden.size())
        # print('encoder_outputs.size',encoder_outputs.size())
        # print('encoder_hidden.size', encoder_hidden.size())
        result1 = self.decoder1(inputs=target_variable1,
                                encoder_hidden=encoder_hidden1,
                                encoder_outputs=encoder_outputs1,
                                function=self.decode_function,
                                teacher_forcing_ratio=teacher_forcing_ratio)
        result2 = self.decoder2(inputs=target_variable2,
                                encoder_hidden=encoder_hidden1,
                                encoder_outputs=encoder_outputs1,
                                function=self.decode_function,
                                teacher_forcing_ratio=teacher_forcing_ratio)
        return result1, result2
