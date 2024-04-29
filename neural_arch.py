import torch

import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score

class F1_Loss_Fn(nn.Module):
    def __init__(self):
        super(F1_Loss_Fn, self).__init__()
        self.sigmoid = nn.Sigmoid()  
    def forward(self, inputs, targets):         
        #for binary class ?
        #inputs = self.sigmoid(inputs[:, 1])   
        #targets = targets[:, 1]   
        
        inputs = self.sigmoid(inputs)  
        intersection = 2.0 * (inputs * targets).sum()
        union = inputs.sum() + targets.sum()
        dice_loss = 1 - (intersection / union)
        return dice_loss


import torch
import torch.nn as nn
import torch.nn.functional as F

#Deep Average Neural Network
class NeuralNetwork(nn.Module):
    def __init__(self, n_embed, d_embed, d_hidden, d_out, embedding_layer):
        super().__init__()
        # Embedding layer setup, ensure it's provided during initialization
        if embedding_layer is None:
            self.emb = nn.Embedding(n_embed, d_embed)
        else:
            self.emb = nn.Embedding.from_pretrained(embedding_layer, freeze=False)

        # Example architecture with text data
        #self.dropout = nn.Dropout(d_out)
        self.fc1 = nn.Linear(d_embed, d_hidden)  
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.fc3 = nn.Linear(d_hidden, d_out)  

    def forward(self, x):
        # Embedding layer with averaging. x should be (batch_size, sequence_length)
        x = self.emb(x)  # (batch_size, sequence_length, d_embed)
        x = torch.mean(x, dim=1)  # Average the embeddings along the sequence (batch_size, d_embed)
        x = F.relu(self.fc1(x))
        #x = self.dropout(x)
        x = F.relu(self.fc2(x))
        #x = self.dropout(x)
        logits = self.fc3(x)
        return logits

 