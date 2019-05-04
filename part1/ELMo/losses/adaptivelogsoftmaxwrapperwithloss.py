import torch
import torch.nn as nn


class AdaptiveLogSoftmaxWrapperWithLoss(nn.AdaptiveLogSoftmaxWithLoss):
    """The wrapper class of AdaptiveLogSoftmaxWithLoss.
    Args:
        kwargs (dict): The parameters for the AdaptiveLogSoftmaxWithLoss class (refer to torch.nn.AdaptiveLogSoftmaxWithLoss).
    """
    def __init__(self, **kwargs):
        super(AdaptiveLogSoftmaxWrapperWithLoss, self).__init__(**kwargs)
        
    def forward(self, input, target, word_lens):
        """
        Args:
            input (tensor) (batch, padded_len, dim_projection):
            target (LongTensor) (batch, padded_len):
            word_lens (LongTensor) (batch, ): The original length of the sequence (input are the same as target).
        Returns:
            
        """
        batch, padded_len, dim_projection = input.size()
        
        """
        ASMoutput (namedtuple)
            output ():
            loss ():
        """
        ASMoutput = super(AdaptiveLogSoftmaxWrapperWithLoss, self).forward(input.reshape(-1, dim_projection), target.reshape(-1))
        output = ASMoutput.output.reshape(batch, padded_len) # output: (batch, padded_len)
        
        #
        mask = torch.ones_like(output, requires_grad=False)
        for i in range(batch):
            mask[i, word_lens[i]:] = 0
        output = output * mask
        
        # the cross entropy loss
        loss = (-output).sum(dim=-1) / word_lens.float()
        return loss