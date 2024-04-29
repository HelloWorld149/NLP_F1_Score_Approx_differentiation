# NLP_F1_Score_Approx_differentiation
light neural netowork with approximate f1 score loss function

# Dataset
SMS dataset: https://www.kaggle.com/datasets/thedevastator/sms-spam-collection-a-more-diverse-dataset
cyberbullying tweets dataset:https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification

# train.py
train.py and train_1.py runs the datasset: sms, cyberbulling respective. cyberbulling dataset is modified to binary class during preprocess.

# neural_arch.py
neural_arch.py contains Approximated F1_Score loss function and deep average neural network for testing an.

# utils.py
utils.py contains other utility function including building vocabulary, map to index, and custom_collate function for the batch.

