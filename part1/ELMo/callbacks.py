import math
import json
import logging


class Callback(object):
    def __init__(self):
        raise NotImplementedError

    def on_epoch_end(self):
        raise NotImplementedError
        
        
class MetricsLogger(Callback):
    def __init__(self, log_dest):
        self.history = {'train': [],
                        'valid': []}
        self.log_dest = log_dest
    
    def on_epoch_end(self, log_train, log_valid, model):
        log_train['epoch'] = model.epoch
        log_valid['epoch'] = model.epoch
        self.history['train'].append(log_train)
        self.history['valid'].append(log_valid)
        with open(self.log_dest, 'w') as f:
            json.dump(self.history, f, indent='    ')


class ModelCheckpoint(Callback):
    def __init__(self, filepath, monitor='loss', mode='min', saved_frequency=1):
        assert mode in ['min', 'max']
        assert saved_frequency > 0
        self.filepath = filepath
        self.monitor = monitor
        self.best = math.inf if mode == 'min' else -math.inf
        self.mode = mode
        self.saved_frequency = saved_frequency
        
    def on_epoch_end(self, log_train, log_valid, model):
        score = log_valid[self.monitor]
        
        if self.mode == 'min':
            if model.epoch % self.saved_frequency == 0:
                model.save('{}.{}.pkl'.format(self.filepath, model.epoch))
                logging.info('model.{} saved'.format(model.epoch))
            if score < self.best:
                self.best = score
                model.not_improved_count = 0
                model.save('{}.best.pkl'.format(self.filepath))
                logging.info('model.best saved (min {}: {})'.format(self.monitor, self.best))
            else:
                model.not_improved_count += 1
                logging.info('model.best remained (min {}: {}) (epoch: {})'.format(self.monitor, 
                                                                                   self.best, 
                                                                                   model.epoch - model.not_improved_count))
                
        elif self.mode == 'max':
            if model.epoch % self.saved_frequency == 0:
                model.save('{}.{}.pkl'.format(self.filepath, model.epoch))
                logging.info('model.{} saved'.format(model.epoch))
            if score > self.best:
                self.best = score
                model.not_improved_count = 0
                model.save('{}.best.pkl'.format(self.filepath))
                logging.info('model.best saved (max {}: {})'.format(self.monitor, self.best))
            else:
                model.not_improved_count += 1
                logging.info('model.best remained (max {}: {}) (epoch: {})'.format(self.monitor, 
                                                                                   self.best, 
                                                                                   model.epoch - model.not_improved_count))