import numpy as np
import pickle
import torch
from ELMo.modules.elmo import ELMo


class Embedder(object):
    """
    The class responsible for loading a pre-trained ELMo model and provide the ``embed``
    functionality for downstream BCN model.

    You can modify this class however you want, but do not alter the class name and the
    signature of the ``embed`` function. Also, ``__init__`` function should always have
    the ``ctx_emb_dim`` parameter.
    """

    def __init__(self, n_ctx_embs, ctx_emb_dim, device, char_vocabulary_path, elmo_model_path, **elmo_model_kwargs):
        """
        The value of the parameters should also be specified in the BCN model config.
        """
        self.n_ctx_embs = n_ctx_embs
        self.ctx_emb_dim = ctx_emb_dim
        self.device = torch.device(f'{device.type}:{device.ordinal}')
        
        with open(char_vocabulary_path, 'rb') as f:
            self.char_vocabulary = pickle.load(f)
            
        self.elmo_model = ELMo(**elmo_model_kwargs)
        self.elmo_model.to(self.device)
        checkpoint = torch.load(elmo_model_path, map_location='cuda:0')
        self.elmo_model.load_state_dict(checkpoint['model'])
        self.elmo_model.eval()
    
    def __call__(self, sentences, max_sent_len):
        """
        Generate the contextualized embedding of tokens in ``sentences``.

        Parameters
        ----------
        sentences : ``List[List[str]]``
            A batch of tokenized sentences.
        max_sent_len : ``int``
            All sentences must be truncated to this length.

        Returns
        -------
        ``np.ndarray``
            The contextualized embedding of the sentence tokens.

            The ndarray shape must be
            ``(len(sentences), min(max(map(len, sentences)), max_sent_len), self.n_ctx_embs, self.ctx_emb_dim)``
            and dtype must be ``np.float32``.
        """
        # transform the sentences to the character index
        processed_sentences = []
        for sentence in sentences:
            sentence = ['<BOS>'] + sentence + ['<EOS>']
            processed_sentence = []
            for word in sentence:
                chars = [word] if word in ['<BOS>', '<EOS>'] else list(word)
                processed_sentence.append([self.char_vocabulary.get(char, self.char_vocabulary['<unk>']) for char in chars])
            processed_sentences.append(processed_sentence)
        
        sentence_lens = torch.tensor([min(len(processed_sentence), max_sent_len + 2)
                                      for processed_sentence in processed_sentences])
        max_sent_len = sentence_lens.max().item()
        word_lens = [len(word)
                     for processed_sentence in processed_sentences 
                     for word in processed_sentence]
        max_word_len = max(word_lens)
        
        # use the ELMo model to extract the features
        forward_input = torch.tensor([input_pad_to_len(processed_sentence, 
                                                       max_sent_len, 
                                                       max_word_len)
                                      for processed_sentence in processed_sentences], dtype=torch.long)
        backward_input = forward_input.flip(dims=[1])        
        with torch.no_grad():
            elmo_features = self.elmo_model.extract_features(forward_input.to(self.device), 
                                                             backward_input.to(self.device), 
                                                             sentence_lens.to(self.device))
        
        # remove the first ('<BOS>') and the last word ('<EOS>' or the 'max_sent_len + 1' word or the padding)
        max_sent_len -= 2
        char_embedding_features = elmo_features.char_embedding[:, 1:max_sent_len+1, :]
        lm_layer0_features = elmo_features.lm_layer0[:, 1:max_sent_len+1, :]
        lm_layer1_features = elmo_features.lm_layer1[:, 1:max_sent_len+1, :]
        return torch.stack([char_embedding_features, lm_layer0_features, lm_layer1_features], dim=2).cpu().numpy()
        #return torch.cat([char_embedding_features, lm_layer0_features, lm_layer1_features], dim=-1).unsqueeze(dim=2).cpu().numpy()
    
def input_pad_to_len(words, padded_word_len, padded_char_len, word_padding=0, char_padding=0):
        """Pad words to 'padded_word_len' with padding if 'len(words) < padded_word_len'. If if 'len(words) > padded_word_len', truncate words to `padded_word_len`. Then add padding so that each word has the same length 'padded_char_len'.
        Example:
            input_pad_to_len([[1, 2, 3],
                              [2, 3],
                              [3]], 5, -1, 0)

            ==============>  [[1, 2, 3],
                              [2, 3, 0],
                              [3, 0, 0],
                              [-1, 0, 0],
                              [-1, 0, 0]]
        Args:
            words (list): List of list of the word index.
            padded_word_len (int): The length for padding a batch of sequences to the same length.
            padded_char_len (int): The length for padding each word in the sentences to the same length.
            word_padding (int): The index used to pad the input sequence.
            char_padding (int): The index used to pad each word in the input sequence.
        """
        if len(words) < padded_word_len:
            words += [[word_padding]] * (padded_word_len - len(words))
        elif len(words) > padded_word_len:
            words = words[:padded_word_len]
        else:
            pass
        words = [word + [char_padding] * (padded_char_len - len(word)) if len(word) < padded_char_len else word for word in words]
        return words