import torch
from torch.utils.data import DataLoader
import torch.utils.data.dataloader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
import logging
import math
import random
import numpy as np


class BasePredictor(object):
    def __init__(self,
                 training=True,
                 valid=None,
                 device=None,
                 metrics={},
                 batch_size=64,
                 max_epochs=10,
                 max_iters_per_train_epoch=2000,
                 learning_rate=1e-3,
                 weight_decay=0,
                 early_stopping=0,
                 grad_clipping=0):
        self.training = training
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.max_iters_per_train_epoch = max_iters_per_train_epoch
        self.valid = valid
        self.metrics = metrics
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.shuffled_indexes = None
        
        if device is not None:
            self.device = torch.device('cuda:{}'.format(device) if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if self.device == torch.device('cpu'):
            logging.warning('Not using GPU.')
            
        if training:
            assert early_stopping >= 0, 'The value of early stopping needs to be greater than zero.'
            if early_stopping == 0:
                logging.warning('Not using early stopping.')
            else:
                logging.info('Using early stopping.')
            self.early_stopping = early_stopping if early_stopping else math.inf

            assert grad_clipping >= 0, 'The value of gradient clipping needs to be greater than zero.'
            if grad_clipping == 0:
                logging.warning('Not using gradient clipping.')
            else:
                logging.info('Using gradient clipping.')
            self.grad_clipping = grad_clipping
            self.epoch = 1
            self.not_improved_count = 0
        
    def fit_dataset(self, dataset, collate_fn=default_collate, callbacks=[]):
        # create the training dataloader, if 'self.max_iters_per_train_epoch > 0', create the training dataloader for every epoch to iterate the whole training set
        if self.max_iters_per_train_epoch > 0:
            train_data = np.array(dataset.data)
            if self.shuffled_indexes is None:
                self.shuffled_indexes = random.sample(population=list(range(len(train_data))), k=len(train_data))
            shuffled_train_data = list(train_data[self.shuffled_indexes])
            samples_per_train_epoch = self.max_iters_per_train_epoch * self.batch_size
            multiples = len(train_data) // samples_per_train_epoch
        else:
            train_dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
        
        # create the validation dataloader
        if self.valid is not None:
            valid_dataloader = DataLoader(dataset=self.valid, batch_size=self.batch_size, shuffle=False, num_workers=4, collate_fn=self.valid.collate_fn)
        else:
            log_valid = None
            
        # Start the training loop.
        while self.epoch <= self.max_epochs and self.not_improved_count < self.early_stopping:
            print()
            logging.info(f'Epoch {self.epoch}')
            
            # 
            if self.max_iters_per_train_epoch > 0:
                remainder = (self.epoch - 1) % multiples
                if self.epoch != 1 and remainder == 0:
                    #shuffled_train_data = random.sample(population=train_data, k=len(train_data))
                    self.shuffled_indexes = random.sample(population=list(range(len(train_data))), k=len(train_data))
                    shuffled_train_data = list(train_data[self.shuffled_indexes])
                dataset.data = shuffled_train_data[samples_per_train_epoch * remainder : samples_per_train_epoch * (remainder + 1)]
                train_dataloader = DataLoader(dataset=dataset, 
                                              batch_size=self.batch_size, 
                                              shuffle=False, # data are shuffled
                                              num_workers=4, 
                                              collate_fn=collate_fn)
                
            log_train = self._run_epoch(train_dataloader, training=True)
            if self.valid is not None:
                log_valid = self._run_epoch(valid_dataloader, training=False)
            
            # evaluate valid score
            for callback in callbacks:
                callback.on_epoch_end(log_train, log_valid, self)
            self.epoch += 1
            
        if self.not_improved_count == self.early_stopping:
            logging.info('Early stopping')
        
    def predict_dataset(self, data,
                        collate_fn=default_collate,
                        batch_size=None,
                        predict_fn=None):
        if batch_size is None:
            batch_size = self.batch_size
        if predict_fn is None:
            predict_fn = self._predict_batch

        # set model to eval mode
        self.model.eval()

        # make dataloader
        dataloader = DataLoader(dataset=data, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

        ys_ = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                batch_y_ = predict_fn(batch)
                ys_.append(batch_y_)

        ys_ = torch.cat(ys_, 0)

        return ys_

    def save(self):
        raise NotImplementedError
        
    def load(self):        
        raise NotImplementedError

    def _run_epoch(self, dataloader, training):
        # set model training/evaluating mode
        self.model.train(training)

        # run batches for train
        total_loss = 0

        # reset metric accumulators
        for metric in self.metrics:
            metric.reset()
        
        # run batches
        iters_in_epoch = len(dataloader)
        trange = tqdm(dataloader, 
                      total=iters_in_epoch, 
                      desc='training' if training else 'evaluating')
        for batch in trange:
            if training:
                output, batch_loss = self._run_iter(batch)
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
            else:
                with torch.no_grad():
                    output, batch_loss = self._run_iter(batch)

            # accumulate loss and metric scores
            total_loss += batch_loss.item()
            for metric in self.metrics:
                metric.update(output)
            trange.set_postfix(loss=batch_loss.item(), **{m.name: m.get_iter_score() for m in self.metrics})
            
        # calculate average loss and metrics
        epoch_log = {}
        epoch_log['loss'] = total_loss / iters_in_epoch
        logging.info(f'loss: {total_loss / iters_in_epoch}')
        for metric in self.metrics:
            score = metric.get_epoch_score()
            logging.info(f'{metric.name}: {score}')
            epoch_log[metric.name] = score
        return epoch_log

    def _run_iter(self):
        """ Run iteration for training.
        """
        raise NotImplementedError

    def _predict_batch(self):
        """ Run iteration for predicting.
        """
        raise NotImplementedError