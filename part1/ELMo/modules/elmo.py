import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from collections import namedtuple
from ELMo.modules.char_embedding import CharEmbedding


class ELMo(nn.Module):
    """Implement the Embeddings from Language Models (ELMo) as described in "Deep contextualized word representations" (https://arxiv.org/pdf/1802.05365.pdf)
        Args:
            hidden_size (int): The number of features in the hidden state h of the language models.
            dim_projection (int):
            char_embedding_kwargs (dict): The parameters for the CharEmbedding class (refer to modules/char_embedding.py).
    """
    def __init__(self, hidden_size=2048, dim_projection=512, **char_embedding_kwargs):
        super(ELMo, self).__init__()
        self.char_embedding = CharEmbedding(**char_embedding_kwargs)
        
        """
        self.output_layer = nn.Sequential(nn.Linear(dim_projection, dim_projection),
                                          nn.ReLU(inplace=True))
        """
        
        # forward language model
        self.forward_lm = nn.Sequential()
        self.forward_lm.add_module('lstm0', nn.LSTM(input_size=char_embedding_kwargs['projection_size'],
                                                    hidden_size=hidden_size,
                                                    num_layers=1,
                                                    dropout=0,
                                                    bidirectional=False))
        self.forward_lm.add_module('linear0', nn.Linear(hidden_size, dim_projection))
        self.forward_lm.add_module('lstm1', nn.LSTM(input_size=dim_projection,
                                                    hidden_size=hidden_size,
                                                    num_layers=1,
                                                    dropout=0,
                                                    bidirectional=False))
        self.forward_lm.add_module('linear1', nn.Linear(hidden_size, dim_projection))
        
        for name, param in self.forward_lm.named_parameters():
            if 'lstm' in name and param.requires_grad:
                # orthogonal initialization
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                    
                # bias = [b_ig | b_fg | b_gg | b_og], set b_fg (forget gate) to 1 and other gates to 0
                elif 'bias' in name:
                    n = param.size(0)
                    param.data.fill_(0)
                    param.data[n // 4 : n // 2].fill_(1)
        
        # backward language model
        self.backward_lm = nn.Sequential()
        self.backward_lm.add_module('lstm0', nn.LSTM(input_size=char_embedding_kwargs['projection_size'],
                                                     hidden_size=hidden_size,
                                                     num_layers=1,
                                                     dropout=0,
                                                     bidirectional=False))
        self.backward_lm.add_module('linear0', nn.Linear(hidden_size, dim_projection))
        self.backward_lm.add_module('lstm1', nn.LSTM(input_size=dim_projection,
                                                     hidden_size=hidden_size,
                                                     num_layers=1,
                                                     dropout=0,
                                                     bidirectional=False))
        self.backward_lm.add_module('linear1', nn.Linear(hidden_size, dim_projection))        
        
        for name, param in self.backward_lm.named_parameters():
            if 'lstm' in name and param.requires_grad:
                # orthogonal initialization
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                    
                # bias = [b_ig | b_fg | b_gg | b_og], set b_fg (forget gate) to 1 and other gates to 0
                elif 'bias' in name:
                    n = param.size(0)
                    param.data.fill_(0)
                    param.data[n // 4 : n // 2].fill_(1)
        
    def packed_forward(self, rnn, padded_input, lengths):
        """
        Args:
            rnn: 
            padded_input (tensor) (padded_len, batch, features): The padded input.
            lengths (LongTensor) (batch, ): The original length of the padded_input
        Return:
            padded_output (tensor) (padded_len, batch, features):
        """
        lengths, sorted_indexes = torch.sort(lengths, descending=True) # sorted by descending order
        padded_input = padded_input.index_select(dim=1, index=sorted_indexes)
        packed_input = pack_padded_sequence(input=padded_input, lengths=lengths)
        packed_output, _ = rnn(packed_input)
        padded_output, _ = pad_packed_sequence(sequence=packed_output, padding_value=0)
        unsorted_indexes = torch.argsort(sorted_indexes) # recover the original order
        return padded_output.index_select(dim=1, index=unsorted_indexes)
    
    def forward(self, forward_input, backward_input, word_lens):
        """
        Args:
            forward_input (tensor) (batch, padded_len): 
            word_lens (LongTensor) (batch, ): The original length of the input sentences.
        Returns:
            logits (dict): 
                forward (tensor) (batch, padded_len, dim_projection): 
                backward (tensor) (batch, padded_len, dim_projection): 
        """        
        forward_char_embedding_features = self.char_embedding(forward_input).transpose(1, 0) # (padded_len, batch, projection_size)
        backward_char_embedding_features = self.char_embedding(backward_input).transpose(1, 0) # (padded_len, batch, projection_size)
        
        forward_lm_layer0_features = self.forward_lm.linear0(self.packed_forward(self.forward_lm.lstm0, forward_char_embedding_features, word_lens)) # (padded_len, batch, projection_size)
        backward_lm_layer0_features = self.backward_lm.linear0(self.packed_forward(self.backward_lm.lstm0, backward_char_embedding_features, word_lens)) # (padded_len, batch, projection_size)
        
        forward_lm_layer1_features = self.forward_lm.linear1(self.packed_forward(self.forward_lm.lstm1, forward_lm_layer0_features, word_lens)) # (padded_len, batch, projection_size)
        backward_lm_layer1_features = self.backward_lm.linear1(self.packed_forward(self.backward_lm.lstm1, backward_lm_layer0_features, word_lens)) # (padded_len, batch, projection_size)
        """
        # residual connection
        forward_lm_layer1_features = forward_lm_layer0_features + forward_lm_layer1_features
        backward_lm_layer1_features = backward_lm_layer0_features + backward_lm_layer1_features
        
        #        
        forward_logits = self.output_layer(forward_lm_layer1_features)
        backward_logits = self.output_layer(backward_lm_layer1_features)
        """
        # residual connnection between layer0 and layer1
        forward_logits = (forward_lm_layer0_features + forward_lm_layer1_features).transpose(1, 0)
        backward_logits = (backward_lm_layer0_features + backward_lm_layer1_features).transpose(1, 0)
        return {'forward': forward_logits, 'backward': backward_logits}
    
    def concat_features(self, forward_features, backward_features, word_lens):
        padded_len, batch, _ = backward_features.size()
        indexes = list(range(padded_len))
        for i in range(batch):
            reversed_indexes = indexes[:word_lens[i]][::-1] + indexes[word_lens[i]:]
            backward_features[:, i, :] = backward_features[:, i, :].index_select(dim=0, 
                                                                                 index=torch.tensor(reversed_indexes, 
                                                                                                    dtype=torch.long,
                                                                                                    device=word_lens.device))
        return torch.cat([forward_features, backward_features], dim=-1)
            
    def extract_features(self, forward_input, backward_input, word_lens):
        """
        Args:
            forward_input (tensor) (batch, padded_len): 
            word_lens (LongTensor) (batch, ): The original length of the input sentences.
        Returns:
            logits (dict): 
                forward (tensor) (batch, padded_len, dim_projection): 
                backward (tensor) (batch, padded_len, dim_projection): 
        """
        forward_char_embedding_features = self.char_embedding(forward_input).transpose(1, 0) # (padded_len, batch, projection_size)
        backward_char_embedding_features = self.char_embedding(backward_input).transpose(1, 0) # (padded_len, batch, projection_size)
        
        forward_lm_layer0_features = self.forward_lm.linear0(self.packed_forward(self.forward_lm.lstm0, forward_char_embedding_features, word_lens)) # (padded_len, batch, projection_size)
        backward_lm_layer0_features = self.backward_lm.linear0(self.packed_forward(self.backward_lm.lstm0, backward_char_embedding_features, word_lens)) # (padded_len, batch, projection_size)
        
        forward_lm_layer1_features = self.forward_lm.linear1(self.packed_forward(self.forward_lm.lstm1, forward_lm_layer0_features, word_lens)) # (padded_len, batch, projection_size)
        backward_lm_layer1_features = self.backward_lm.linear1(self.packed_forward(self.backward_lm.lstm1, backward_lm_layer0_features, word_lens)) # (padded_len, batch, projection_size)
        
        # concatenate forward and backward features
        char_embedding_features = self.concat_features(forward_char_embedding_features, 
                                                       backward_char_embedding_features,
                                                       word_lens).transpose(1, 0) # (batch, padded_len, 2 * projection_size)
        lm_layer0_features = self.concat_features(forward_lm_layer0_features, 
                                                  backward_lm_layer0_features,
                                                  word_lens).transpose(1, 0) # (batch, padded_len, 2 * projection_size)
        lm_layer1_features = self.concat_features(forward_lm_layer1_features, 
                                                  backward_lm_layer1_features,
                                                  word_lens).transpose(1, 0) # (batch, padded_len, 2 * projection_size)
        Features = namedtuple('ELMoFeatures', ['char_embedding', 'lm_layer0', 'lm_layer1'])
        return Features(*[char_embedding_features, lm_layer0_features, lm_layer1_features])