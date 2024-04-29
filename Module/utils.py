from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from nltk.tokenize import word_tokenize
import torch

#compute metric
from sklearn.metrics import f1_score, accuracy_score

def compute_metrics(preds, labels):
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)  
    return acc, f1

def custom_collate(batch):
    sentence = [item['sentence'] for item in batch]
    label = torch.stack([item['label'] for item in batch])
    sentence = pad_sequence(sentence, batch_first=True, padding_value=0)
    return {
        'sentence': sentence,
        'label': label
    }

def sentence_to_indices(sentence, vocab):
    tokens = word_tokenize(sentence.lower())
    unknown_index = vocab.get('<unk>')
    indices = [vocab.get(token, unknown_index) for token in tokens]
    
    return indices

class MyDataset(Dataset):
    def __init__(self, X, y, vocab):
        self.text = X
        self.labels = y
        self.vocab = vocab
        self.num_classes = y.nunique()
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        try:
            sentence = self.text['sms'].iloc[idx]
        except:
            sentence = self.text['tweet_text'].iloc[idx]
        sentence = torch.tensor(sentence_to_indices(sentence, self.vocab), dtype=torch.long)
        label = torch.tensor(self.labels.iloc[idx], dtype=torch.long)
        label = torch.nn.functional.one_hot(label, num_classes=self.num_classes).float()
        
                
        return {
            'sentence': sentence,
            'label': label
        }
