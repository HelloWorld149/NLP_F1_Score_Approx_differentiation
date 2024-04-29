import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import random
random.seed(577)

import numpy as np
np.random.seed(577)
import pandas as pd

import torch
torch.set_default_tensor_type(torch.FloatTensor)
torch.use_deterministic_algorithms(True)
torch.manual_seed(577)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch.optim as optim

from neural_arch import F1_Loss_Fn, NeuralNetwork, DAN


import gensim.downloader as api


from torch.utils.data import DataLoader, random_split


from Module.utils import MyDataset, custom_collate, compute_metrics

import torch.nn as nn

import copy

import warnings  
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    print("CUBLAS_WORKSPACE_CONFIG:", os.getenv("CUBLAS_WORKSPACE_CONFIG"))
    #Read dataset and then preprocess
    cyber_bulling_df = pd.read_csv('dataset\cyberbullying_tweets.csv')
    cyber_bulling_df['label'] = np.where(
        cyber_bulling_df['label'] == "not_cyberbullying", 0, 1)
    sms_df = pd.read_csv('dataset\sms_dataset.csv')
    X = cyber_bulling_df.drop('label', axis=1)
    y = cyber_bulling_df['label']
    
    # call glove-wiki and make vocaburary 
    glove_embs = api.load("glove-wiki-gigaword-50")
    vocab = {word: index for index, word in enumerate(glove_embs.index_to_key)}
    emb_weights = torch.FloatTensor(glove_embs.vectors) 
    unknown_vector = torch.zeros(1, emb_weights.size(1))  #zero vector for unknown vector
    emb_weights = torch.cat([emb_weights, unknown_vector], 0)
    unknown_index = emb_weights.size(0) - 1  # Index of the unknown vector
    vocab['<unk>'] = unknown_index  #update vocab with unknown index
    embedding_layer = torch.nn.Embedding.from_pretrained(emb_weights, freeze=False)
    n_embed, d_embed = emb_weights.size()  #embed size
    
    model = NeuralNetwork(n_embed= n_embed, d_embed= d_embed, d_hidden=512, d_out=2, embedding_layer= emb_weights)
    # loss function and optimizer
    loss_fn = F1_Loss_Fn()  # F1_apprx_loss fn
    loss_fn_1 = nn.CrossEntropyLoss()  #Cross Entropy Loss function
    loss_fn_2 = nn.MSELoss()  #MSE Loss function
    optimizer = optim.Adam(model.parameters(), lr=0.00005)
    
    # Load the datasets and set train size and test size
    full_dataset = MyDataset(X, y, vocab)
    total_size = len(full_dataset)
    train_size = int(0.6 * total_size)
    val_size = int(0.2 * total_size)  # Explicitly define validation size here
    test_size = total_size - train_size - val_size  # Ensure all data is accounted for

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate)
    
    num_epochs = 15
    
    best_val_loss = float('inf')  # Initialize best validation loss as infinity
    best_model_weights = copy.deepcopy(model.state_dict())  # Copy of the initial model weights
    
    
    
    for epoch in range(num_epochs):
        num_batches = 0
        model.train()
        total_loss = 0
        total_acc = 0
        total_f1 = 0
        
        #================Begin Train========================
        model.train()
        for batch in train_loader:
            sentence = batch['sentence']
            label = batch['label']
            y_pred = model(sentence)
            loss = loss_fn(y_pred, label)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            total_loss += loss.item()
            labels = label.argmax(dim=1)
            preds = y_pred.argmax(dim=1)
            acc, f1 = compute_metrics(preds.cpu(), labels.cpu())
            total_acc += acc
            total_f1 += f1

        num_batches = len(train_loader)
        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
        avg_f1 = total_f1 / num_batches
        print(f'Train Epoch {epoch}, Loss: {avg_loss}, Accuracy: {avg_acc}, F1 Score: {avg_f1}')

        #==========================Begin Validation==================================
        model.eval()
        
        total_loss = 0
        total_acc = 0
        total_f1 = 0
        
        for batch in val_loader:
            sentence = batch['sentence']
            label = batch['label']
            y_pred = model(sentence)
            loss = loss_fn_1(y_pred, label)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            total_loss += loss.item()
            labels = label.argmax(dim=1)
            preds = y_pred.argmax(dim=1)
            acc, f1 = compute_metrics(preds.cpu(), labels.cpu())
            total_acc += acc
            total_f1 += f1

        num_batches = len(val_loader)
        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
        avg_f1 = total_f1 / num_batches
        print(f'validation Epoch {epoch}, Loss: {avg_loss}, Accuracy: {avg_acc}, F1 Score: {avg_f1}')
        
        # Update best model weights if current model is better
        if avg_loss < best_val_loss:
            print(f'Validation loss improved from {best_val_loss} to {avg_loss}. Saving best model...')
            best_val_loss = avg_loss
            best_model_weights = copy.deepcopy(model.state_dict())  # Update best model weights
    
    #===================Begin Test========================
    model.eval()  
    true_labels = []
    predictions = []
    
    # After training is complete, load the best model weights
    model.load_state_dict(best_model_weights)

    with torch.no_grad():
        for batch in test_loader:
            sentence = batch['sentence']
            label = batch['label']
            outputs = model(sentence)
            labels = label.argmax(dim=1)
            preds = outputs.argmax(dim=1)
            true_labels.extend(labels.cpu().numpy())
            predictions.extend(preds.cpu().numpy())

    acc, f1 = compute_metrics(predictions, true_labels)

    # Calculate accuracy and F1 score
    acc, f1 = compute_metrics(predictions, true_labels)
    print(f"Final Accuracy for the test set: {acc}")
    print(f"Final F1 Score for the test set: {f1}")

    # Write predictions to test.pred.txt (optional, based on specific requirements)
    output_file_path = 'test.pred.txt'
    with open(output_file_path, 'w') as file:
        for prediction in predictions:
            file.write(f"{prediction}\n")

    

        

        
        

    
    
    
    