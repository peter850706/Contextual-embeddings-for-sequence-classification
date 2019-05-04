import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertForSequenceClassification


class BertForSequenceClassificationWrapper(BertForSequenceClassification):
    def __init__(self, config, num_labels, dropout):
        super(BertForSequenceClassificationWrapper, self).__init__(config, num_labels)
        del self.classifier
        del self.dropout
        self.classifier = nn.Sequential(nn.Dropout(dropout),
                                        nn.Linear(config.hidden_size * 4, num_labels))
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        encoded_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)        
        
        # concatenate the last four hidden layers' features.        
        pooled_output = torch.cat(encoded_layers[-4:], dim=-1)[:, 0, :]
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits