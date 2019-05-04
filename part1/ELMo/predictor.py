import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
from ELMo.base_predictor import BasePredictor
import math
import logging
import os


def get_instance(module, config, *args, **kwargs):
    return getattr(module, config['type'])(*args, **config['kwargs'], **kwargs)


class Predictor(BasePredictor):
    """Predictor
    Args:
        arch (dict): The parameters to define model architecture.
        loss (dict): The parameters to define loss.
    """
    def __init__(self, arch=None, loss=None, **kwargs):
        super(Predictor, self).__init__(**kwargs)        
        assert arch is not None and isinstance(arch, dict)
        
        module_arch = importlib.import_module("ELMo.modules." + arch['type'].lower())
        self.model = get_instance(module_arch, arch)
        self.model = self.model.to(self.device)
        
        if self.training:
            # define loss
            assert loss is not None and isinstance(loss, dict)
            module_loss = importlib.import_module("ELMo.losses." + loss['type'].lower())
            self.loss = get_instance(module_loss, loss)
            self.loss = self.loss.to(self.device)
            
            # make optimizer
            model_parameters = list(filter(lambda p: p.requires_grad, self.model.parameters()))
            loss_parameters = list(filter(lambda p: p.requires_grad, self.loss.parameters()))
            self.optimizer = torch.optim.Adam([{'params': model_parameters}, {'params': loss_parameters}], lr=self.learning_rate, weight_decay=self.weight_decay)
            
            # gradient clipping
            if self.grad_clipping != 0:
                for param_group in self.optimizer.param_groups:
                    for param in param_group['params']:
                        param.register_hook(lambda x: x.clamp(min=-self.grad_clipping, max=self.grad_clipping))
        
    def save(self, path):
        torch.save({'epoch': self.epoch,
                    'model': self.model.state_dict(),
                    'loss': self.loss.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'shuffled_indexes': self.shuffled_indexes}, path)
        
    def load(self, path):
        checkpoint = torch.load(path, map_location='cuda:0')
        if self.training:
            self.epoch = checkpoint['epoch'] + 1
            self.model.load_state_dict(checkpoint['model'])
            self.loss.load_state_dict(checkpoint['loss'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.shuffled_indexes = checkpoint['shuffled_indexes']
        else:
            self.model.load_state_dict(checkpoint['model'])
            
    def _run_iter(self, batch):
        """Run iteration for training.
        Args:
            batch (dict)
        Returns:
            predicts: Prediction of the batch.
            loss (FloatTensor): Loss of the batch.
        """
        logits = self.model(batch['forward']['input'].to(self.device), 
                            batch['backward']['input'].to(self.device), 
                            batch['word_lens'].to(self.device))
        forward_loss = self.loss(logits['forward'], 
                                 batch['forward']['target'].to(self.device),
                                 batch['word_lens'].to(self.device))
        backward_loss = self.loss(logits['backward'], 
                                  batch['backward']['target'].to(self.device),
                                  batch['word_lens'].to(self.device))
        return torch.cat([forward_loss, backward_loss], dim=0), (forward_loss.mean() + backward_loss.mean()) / 2

    def _predict_batch(self, batch):
        """Run iteration for predicting.
        Args:
            batch (dict)
        Returns:
            predicts: Prediction of the batch.
        """
        
        return logits