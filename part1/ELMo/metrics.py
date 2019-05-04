import torch


class Metrics(object):
    def __init__(self):
        self.name = 'metric name'

    def reset(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    def get_iter_score(self):
        raise NotImplementedError
    
    def get_epoch_score(self):
        raise NotImplementedError


class Perplexity(Metrics):
    """
    Args:
         
    """
    def __init__(self):
        self.name = 'perplexity'
        self.n = 0
        self.perplexity = 0
        self.iter_score = 0
        
    def reset(self):
        self.n = 0
        self.perplexity = 0
    
    def update(self, loss):
        """
        Args:
            loss (tensor) (batch, ): The negative log likelihood loss.
        """
        n = loss.size(0)
        perplexity = loss.exp().sum().item()
        self.n += n
        self.perplexity += perplexity
        self.iter_score = perplexity / n
    
    def get_iter_score(self):        
        return self.iter_score  ##'{:.3f}'.format(self.get_score())
                
    def get_epoch_score(self):
        return self.perplexity / self.n