from __future__ import division
import logging
import os
import random
import time

import torch
import torchtext
from torch import optim

import seq2seq
from seq2seq.evaluator import Evaluator
from seq2seq.loss import NLLLoss
from seq2seq.optim import Optimizer
from seq2seq.util.checkpoint import Checkpoint

class SupervisedTrainer(object):
    """ The SupervisedTrainer class helps in setting up a training framework in a
    supervised setting.

    Args:
        expt_dir (optional, str): experiment Directory to store details of the experiment,
            by default it makes a folder in the current directory to store the details (default: `experiment`).
        loss (seq2seq.loss.loss.Loss, optional): loss for training, (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for experiment, (default: 64)
        checkpoint_every (int, optional): number of batches to checkpoint after, (default: 100)
    """
    def __init__(self, loss1=NLLLoss(), loss2=NLLLoss(), loss3=NLLLoss(), batch_size=64,
                 random_seed=None,
                 checkpoint_every=100, print_every=100, expt_dir='experiment',):
        self._trainer = "Simple Trainer"
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)
        self.loss1 = loss1
        self.loss2 = loss2
        self.loss3 = loss3
        self.evaluator = Evaluator(loss1=self.loss1, loss2=self.loss2, loss3=self.loss3, batch_size=batch_size)
        self.optimizer = None
        self.checkpoint_every = checkpoint_every
        self.print_every = print_every

        if not os.path.isabs(expt_dir):
            expt_dir = os.path.join(os.getcwd(), expt_dir)
        self.expt_dir = expt_dir
        if not os.path.exists(self.expt_dir):
            os.makedirs(self.expt_dir)
        self.batch_size = batch_size

        self.logger = logging.getLogger(__name__)

    def get_model_out(self, input_variable1,  input_lengths1, target_variable1,
                      target_variable2, target_variable3, model, teacher_forcing_ratio):
        d1, d2, d3 = model(input_variable1, input_lengths1, target_variable1, target_variable2, target_variable3,
                           teacher_forcing_ratio=teacher_forcing_ratio)
        return d1, d2, d3

    def _train_batch(self, d1, target_variable1, model, loss1):
        decoder_outputs1, decoder_hidden1, other1 = d1
        # Get loss1
        loss1.reset()
        for step, step_output in enumerate(decoder_outputs1):
            batch_size = target_variable1.size(0)
            loss1.eval_batch(step_output.contiguous().view(batch_size, -1), target_variable1[:, step + 1])
        # Backward propagation
        model.zero_grad()
        loss1.backward()
        self.optimizer.step()
        return loss1.get_loss()

    def _train_epoches(self, data1, model, n_epochs, start_epoch, start_step,
                       dev_data1, teacher_forcing_ratio=0):
        log = self.logger

        print_loss1_total = 0  # Reset every print_every
        epoch_loss1_total = 0  # Reset every epoch
        print_loss2_total = 0  # Reset every print_every
        epoch_loss2_total = 0  # Reset every epoch
        print_loss3_total = 0  # Reset every print_every
        epoch_loss3_total = 0  # Reset every epoch

        device = None if torch.cuda.is_available() else -1
        batch_iterator1 = torchtext.data.BucketIterator(
            dataset=data1, batch_size=self.batch_size,
            sort=False, sort_within_batch=True,
            sort_key=lambda x: len(x.bl),
            device=device, repeat=False)

        steps_per_epoch = len(batch_iterator1)
        total_steps = steps_per_epoch * n_epochs

        step = start_step
        step_elapsed = 0
        # loss1 = loss2 = 0
        for epoch in range(start_epoch, n_epochs + 1):
            log.debug("Epoch: %d, Step: %d" % (epoch, step))

            batch_generator1 = batch_iterator1.__iter__()
            for _ in range((epoch - 1) * steps_per_epoch, step):
                next(batch_generator1)

            model.train(True)

            # json版数据迭代器
            for batch1 in batch_generator1:
                step += 1
                step_elapsed += 1

                input_variables1, input_lengths1 = getattr(batch1, seq2seq.bl_field_name)
                # input_variables2, input_lengths2 = getattr(batch1, seq2seq.jz_field_name)

                target_variables1 = getattr(batch1, seq2seq.label_field_name1)
                target_variables2 = getattr(batch1, seq2seq.label_field_name2)
                target_variables3 = getattr(batch1, seq2seq.label_field_name3)

                if torch.cuda.is_available():
                    input_variables1 = input_variables1.cuda()
                    target_variables1 = target_variables1.cuda()
                    target_variables2 = target_variables2.cuda()
                    target_variables3 = target_variables3.cuda()

                # i = random.randint(0, 1)
                # if i == 0:
                d1, d2, d3 = self.get_model_out(input_variables1, input_lengths1, target_variables1, target_variables2,
                                                target_variables3, model, teacher_forcing_ratio)
                decoder_outputs1, decoder_hidden1, other1 = d1
                decoder_outputs2, decoder_hidden2, other1 = d2
                decoder_outputs3, decoder_hidden3, other3 = d3
                # Get loss1
                loss1 = self.loss1
                loss1.reset()
                loss2 = self.loss2
                loss2.reset()
                loss3 = self.loss3
                loss3.reset()
                for step, step_output in enumerate(decoder_outputs1):
                    batch_size = target_variables1.size(0)
                    loss1.eval_batch(step_output.contiguous().view(batch_size, -1), target_variables1[:, step + 1])
                for step, step_output in enumerate(decoder_outputs2):
                    batch_size = target_variables2.size(0)
                    loss2.eval_batch(step_output.contiguous().view(batch_size, -1), target_variables2[:, step + 1])
                for step, step_output in enumerate(decoder_outputs3):
                    batch_size = target_variables3.size(0)
                    loss3.eval_batch(step_output.contiguous().view(batch_size, -1), target_variables3[:, step + 1])

                loss1.acc_loss += loss2.acc_loss
                loss1.norm_term += loss2.norm_term
                loss1.acc_loss += loss3.acc_loss
                loss1.norm_term += loss3.norm_term
                # Backward propagation
                model.zero_grad()
                loss1.backward()
                self.optimizer.step()
                loss1 = loss1.get_loss()

                # d1, d2, d3 = self.get_model_out(input_variables1, input_lengths1, target_variables1, target_variables2,
                #                                 target_variables3, model, teacher_forcing_ratio)
                # loss2 = self._train_batch(d2, target_variables2, model, self.loss2)
                #
                # d1, d2, d3 = self.get_model_out(input_variables1, input_lengths1, target_variables1, target_variables2,
                #                                 target_variables3, model, teacher_forcing_ratio)
                # loss3 = self._train_batch(d3, target_variables3, model, self.loss3)

                # Record average loss
                print_loss1_total += loss1
                epoch_loss1_total += loss1

                if step % self.print_every == 0 and step_elapsed > self.print_every:
                    print_loss1_avg = print_loss1_total / self.print_every
                    print_loss1_total = 0
                    log_msg1 = 'Progress: %d%%, Train-loss1 %s: %.4f' % (
                        step / total_steps * 100,
                        self.loss1.name,
                        print_loss1_avg)
                    log.info(log_msg1)

                # print_loss2_total += loss2
                # epoch_loss2_total += loss2
                #
                # if step % self.print_every == 0 and step_elapsed > self.print_every:
                #     print_loss2_avg = print_loss2_total / self.print_every
                #     print_loss2_total = 0
                #     log_msg2 = 'Progress: %d%%, Train-loss2 %s: %.4f' % (
                #         step / total_steps * 100,
                #         self.loss2.name,
                #         print_loss2_avg)
                #     log.info(log_msg2)
                #
                # print_loss3_total += loss3
                # epoch_loss3_total += loss3
                #
                # if step % self.print_every == 0 and step_elapsed > self.print_every:
                #     print_loss3_avg = print_loss3_total / self.print_every
                #     print_loss3_total = 0
                #     log_msg3 = 'Progress: %d%%, Train-loss3 %s: %.4f' % (
                #         step / total_steps * 100,
                #         self.loss3.name,
                #         print_loss3_avg)
                #     log.info(log_msg3)
                # Checkpoint
                if step % self.checkpoint_every == 0 or step == total_steps:
                    Checkpoint(model=model,
                               optimizer=self.optimizer,
                               epoch=epoch, step=step,
                               input_vocab1=data1.fields[seq2seq.bl_field_name].vocab,
                               # input_vocab2=data1.fields[seq2seq.jz_field_name].vocab,
                               output_vocab1=data1.fields[seq2seq.label_field_name1].vocab,
                               output_vocab2=data1.fields[seq2seq.label_field_name2].vocab,
                               output_vocab3=data1.fields[seq2seq.label_field_name3].vocab).save(self.expt_dir)
            if step_elapsed == 0:
                continue
            epoch_loss1_avg = epoch_loss1_total / min(steps_per_epoch, step - start_step)
            epoch_loss1_total = 0
            log_msg1 = "Finished epoch %d: Train-loss1 %s: %.4f" % (epoch, self.loss1.name, epoch_loss1_avg)

            epoch_loss2_avg = epoch_loss2_total / min(steps_per_epoch, step - start_step)
            epoch_loss2_total = 0
            log_msg2 = "Finished epoch %d: Train-loss2 %s: %.4f" % (epoch, self.loss2.name, epoch_loss2_avg)

            epoch_loss3_avg = epoch_loss3_total / min(steps_per_epoch, step - start_step)
            epoch_loss3_total = 0
            log_msg3 = "Finished epoch %d: Train-loss3 %s: %.4f" % (epoch, self.loss3.name, epoch_loss3_avg)

            # 验证数据集
            if dev_data1 is not None:
                dev_loss1, dev_loss2, dev_loss3, accuracy1, accuracy2, accuracy3 = self.evaluator.evaluate(model, dev_data1)

                self.optimizer.update(dev_loss1, epoch)
                self.optimizer.update(dev_loss2, epoch)
                self.optimizer.update(dev_loss3, epoch)
                log_msg1 += ", loss1-Dev %s: %.4f, Accuracy: %.4f" % (self.loss1.name, dev_loss1, accuracy1)
                model.train(mode=True)
                log_msg2 += ", loss2-Dev %s: %.4f, Accuracy: %.4f" % (self.loss2.name, dev_loss2, accuracy2)
                model.train(mode=True)
                log_msg3 += ", loss3-Dev %s: %.4f, Accuracy: %.4f" % (self.loss3.name, dev_loss3, accuracy3)
                model.train(mode=True)

            else:
                self.optimizer.update(epoch_loss1_avg, epoch)
                self.optimizer.update(epoch_loss2_avg, epoch)
                self.optimizer.update(epoch_loss3_avg, epoch)
            log.info(log_msg1)
            log.info(log_msg2)
            log.info(log_msg3)

    def train(self, model, data1, dev_data1, num_epochs=5, resume=False, optimizer=None, teacher_forcing_ratio=0):
        """ Run training for a given model.

        Args:
            model (seq2seq.models): model to run training on, if `resume=True`, it would be
               overwritten by the model loaded from the latest checkpoint.
            data (seq2seq.dataset.dataset.Dataset): dataset object to train on
            num_epochs (int, optional): number of epochs to run (default 5)
            resume(bool, optional): resume training with the latest checkpoint, (default False)
            dev_data (seq2seq.dataset.dataset.Dataset, optional): dev Dataset (default None)
            optimizer (seq2seq.optim.Optimizer, optional): optimizer for training
               (default: Optimizer(pytorch.optim.Adam, max_grad_norm=5))
            teacher_forcing_ratio (float, optional): teaching forcing ratio (default 0)
        Returns:
            model (seq2seq.models): trained model.
        """
        # If training is set to resume
        if resume:
            latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.expt_dir)
            resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
            model = resume_checkpoint.model
            self.optimizer = resume_checkpoint.optimizer

            # A walk around to set optimizing parameters properly
            resume_optim = self.optimizer.optimizer
            defaults = resume_optim.param_groups[0]
            defaults.pop('params', None)
            defaults.pop('initial_lr', None)
            self.optimizer.optimizer = resume_optim.__class__(model.parameters(), **defaults)

            start_epoch = resume_checkpoint.epoch
            step = resume_checkpoint.step
        else:
            start_epoch = 1
            step = 0
            if optimizer is None:
                optimizer = Optimizer(optim.Adam(model.parameters()), max_grad_norm=5)
            self.optimizer = optimizer

        self.logger.info("Optimizer: %s, Scheduler: %s" % (self.optimizer.optimizer, self.optimizer.scheduler))

        self._train_epoches(data1,  model, num_epochs,
                            start_epoch, step, dev_data1,
                            teacher_forcing_ratio=teacher_forcing_ratio)
        return model
