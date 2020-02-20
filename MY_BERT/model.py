import torch
import torch.nn as nn
import numpy as np
import torch
from transformers import  BertModel

class my_model(nn.Module):
    def __init__(self,out_classes):
        super(my_model,self).__init__()
        self.bert_pretrained_model = BertModel.from_pretrained('bert-base-chinese')
        # Set the model in evaluation mode to deactivate the DropOut modules
        # This is IMPORTANT to have reproducible results during evaluation!
        self.bert_pretrained_model.eval()

        self.bert_pretrained_model.to('cuda')

        self.hidden_size= self.bert_pretrained_model.config.hidden_size
        self.tanh=nn.Tanh()
        self.project_layer=nn.Linear(self.hidden_size,out_classes)



    def forward(self, input):

        # Predict hidden states features for each layer
        with torch.no_grad():
            # See the models docstrings for the detail of the inputs
            outputs = self.bert_pretrained_model(input)
            # Transformers models always output tuples.
            # See the models docstrings for the detail of all the outputs
            # In our case, the first element is the hidden state of the last layer of the Bert model
            encoded_layers = outputs[0]

        # We have encoded our input sequence in a FloatTensor of shape (batch size, sequence length, model hidden dimension)

        output=self.project_layer(encoded_layers)
        #(batch size, sequence length, out_class)
        return output
#
# if __name__ == '__main__':
#     #bert_pretrained_model = BertModel.from_pretrained('bert-base-chinese')

