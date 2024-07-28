#!/usr/bin/env python
# coding: utf-8

# ---- LIBRARY IMPORTS ----
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
import time
from datetime import timedelta

# In[1]:
if __name__ == "__main__":

    
    # ---- DEVICE SET UP ----
    def get_gpu():
        if torch.backends.mps.is_available():
            print("Using mps")
            return "mps"
        elif torch.backend.cuda.is_available():
            print("Using cuda")
            return"cuda"
        else:
            print("Using CPU")
            return "cpu"
    device = get_gpu()
    
    
    # clean_data.to_csv('data/fake_news/clean_news_data.csv', index=False)
    dypes = {'text': str, 'label': 'Int64'}
    clean_data = pd.read_csv("data/fake_news/clean_news_data.csv", dtype=dypes)
    # Drop empty strings
    clean_data = clean_data.dropna()
    
    
    # In[3]:
    
    
    # split data into train, test sets
    
    train, test = train_test_split(clean_data, test_size=0.2, random_state=0)
    
    # Create an iterator object for train and test data
    
    data_iter = clean_data.iterrows()
    train_iter = train.iterrows()
    test_iter = test.iterrows()
    
    # Convert generators to list of tuples because DataLoader does not work well with pandas dataframes
    # Use this as inputs for DataLoader
    data_list_of_tuples = [(row.text, row.label) for index, row in data_iter]
    train_list_of_tuples = [(row.text, row.label) for index, row in train_iter]
    test_list_of_tuples = [(row.text, row.label) for index, row in test_iter]
    
    # Taken from pytorch documentation tutorials -> https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
    # tokenization for word sequences
    # No tokenizer is required as data was tokenized in previous step.  We only require tokenizer to split articles by word to create the sequences
    tokenizer = get_tokenizer(tokenizer=None)
    
    def yield_tokens(data):
        # pull the text data from series to tokenize it
        # Each row is a series when calling the iterrows() method, you must call the text column to pull its value
        for index, row in data.iterrows():
            text = row.text
            yield tokenizer(text)
    
    # vocab_dict is now a function that takes a list of words as an input and returns integers based on the indexes found in the vocab_dict's dictionary
    # <unk> -> In case a word is not in vocab_dict, we default it to a special index for words not in vocab_dict
    vocab_dict = build_vocab_from_iterator(iterator=yield_tokens(clean_data), specials=["<unk>"], max_tokens=100)
    vocab_dict.set_default_index(vocab_dict["<unk>"])
    # text_sequencer is a function that takes a string and returns a list of integers based off vocab_dict
    text_sequencer = lambda string: vocab_dict(tokenizer(string))
    
    
    def collate_batch(batch):
        """
        This function takes a batch created from the DataLoader function and does data preprocessing to it
        """
        labels, text_tensors_list = [], []
        for example in batch:
        # Get data from pandas series
            text = example[0]
            label = example[1]
        # convert text to sequences of integers
            text_sequence = text_sequencer(text)
        # convert text_sequence to tensor
            text_sequence_tensor = torch.tensor(text_sequence, dtype=torch.int64)
        # append tensors to lists
            labels.append(label)
            text_tensors_list.append(text_sequence_tensor)
        # add padding of 0 to text_tensors (All articles have a different number of words and we want all tensors to be the same size)
        text_tensors = pad_sequence(text_tensors_list, batch_first=True, padding_value = 0)
        
        # convert labels lists to tensor
        labels_tensor = torch.tensor(labels, dtype=torch.int64)
        return labels_tensor.to(device), text_tensors.to(device)
    
    
    # # Model Building
    
    # ## RNN
    
    # In[4]:
    
    
    class RNN(nn.Module):
        def __init__(self, input_size, hidden_size):
            super(RNN, self).__init__()
            self.hidden_size = hidden_size
        # this converts your sequence of words to a vector to pass through the NN
            self.word_sequence_to_embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=3)
            self.rnn = nn.RNN(3, hidden_size, 1, batch_first=True)
            # self.hidden_layer_input = nn.Linear(in_features=10, out_features=hidden_size, bias=False)
            # self.hidden_layer_previous = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)
            self.hidden_layer_out = nn.Linear(in_features=hidden_size, out_features=1)
            self.activation = nn.Sigmoid()
    
        def forward(self, input_):
            print(input_.shape)
            sequence_embeddings = self.word_sequence_to_embedding(input_)
            print(sequence_embeddings.shape)
            hidden_initial = torch.zeros(1,sequence_embeddings.shape[0], self.hidden_size).to(device)
            # print(sequence_embeddings.shape)
            # hidden = F.tanh(self.hidden_layer_input(sequence_embeddings) + self.hidden_layer_previous(hidden))
            output, hidden = self.rnn(sequence_embeddings, hidden_initial)
            # print(hidden.shape)
            output = self.hidden_layer_out(output)
            print(output.shape)
            output = output[:, -1, :]
            output = self.activation(output)
            print(output.shape)
            return output, hidden
    
        # def initHidden(self):
        # # Return a matrix of 1 row and k columns where k=hidden_size
        #     return torch.zeros(1, self.hidden_size)
            
    
    
    # In[5]:
    
    
    # Initialize Model
    n_hidden = 3
    input_size = len(vocab_dict)
    rnn_model = RNN(input_size=input_size, hidden_size=n_hidden)
    rnn_model.to(device)
    
    
    # ### Training
    
    # In[6]:
    
    
    def get_labels(results):
        decision = lambda val: 1 if val >= 0.5 else 0
        labels = torch.where(results > 0.5, 1.0, 0.0)
        return labels
    
    
    # In[7]:
    
    
    train_loader = torch.utils.data.DataLoader(train_list_of_tuples, batch_size=4, shuffle=True, collate_fn=collate_batch)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(rnn_model.parameters(), lr=0.001, momentum=0.9)
    
    # for input, label in train_loader:
    #     print(input)
    
    
    # In[8]:
    
    
    epoch_iter = 2
    def rnn_train(model, dataloader, nbr_of_epochs,optimizer, criterion):
        for epoch in range(epoch_iter):
            # ---- MODEL TRAINING ----
        # Put model in training mode
            model.train()
        # Start timer for trainig of epoch
            start_time = time.perf_counter()
        # Initalize the batch number currently being worked on
            batch_nbr = 0
        # Create loss variable for epoch
            training_losses = 0.0
        # Total examples labeld correctly in epoch
            training_total_correct = 0.0
        # Total number of examples in epoch
            training_total_examples = 0.0
        
        # Create loss variable for epoch
            testing_losses = 0.0
        # Total examples labeld correctly in epoch
            testing_total_correct = 0.0
        # Total number of examples in epoch
            testing_total_examples = 0.0
        # Execute Forward, Backward, Optimization
            for labels, inputs in train_loader:
            # get the inputs; data is a list of [inputs, labels]
                inputs, labels = inputs.to(device), labels.to(device)
    
            # zero the parameter gradients
                optimizer.zero_grad()
        
        # ---- FORWARD, BACKWARD, OPTIMIZE ----
            # Get model training predictions
                outputs, hidden = model(inputs)
                outputs = torch.squeeze(outputs, 1)
                print(outputs.shape)
                
            # Convert model training predictions to their respective classifications
                predicted_labels = get_labels(outputs)
                
            # Compute Loss of current batch
                loss = criterion(outputs, labels)
            # remove outputs
                del outputs
            # Compute total number of correctly classified examples
                nbr_of_correct_predictions = torch.sum(predicted_labels == labels).item()
            # Grab batch size
                total_nbr_of_elements = labels.shape[0]
            # Compute number of correctly labeled examples and the total exampes
                training_total_correct += nbr_of_correct_predictions
                training_total_examples += total_nbr_of_elements
        
            # Backward pass
                loss.backward()
            # Update model params
                optimizer.step()
            # update training loss of epoch
                training_losses += float(loss)*total_nbr_of_elements
                del loss
            #  update the current batch number of epoch
                batch_nbr += 1
            print("--- COMPLETED ONE EPOCH ----")
        # End training time
        end_time = time.perf_counter()
    # Get total runntime of epoch
        epoch_runtime = timedelta(seconds=end_time-start_time).total_seconds()
        total_training_epoch_loss = round(training_losses/len(train_loader), 4)
    
        result_dict = {"training_loss": total_training_epoch_loss, 'training_accuracy': training_accuracy, "runtime": epoch_runtime}
        print(f'Epoch {epoch + 1}/{epoch_iter} <-> Runtime: {round(epoch_runtime, 0)}s <-> Training loss: {total_training_epoch_loss} <-> Training Accuracy: {training_accuracy} <-> Testing loss: {total_testing_epoch_loss} <-> Testing Accuracy: {testing_accuracy}')
        print('\nTraining Complete')
    
    
    # In[ ]:
    
    
    rnn_train(rnn_model, train_loader,epoch_iter, optimizer, criterion)
    
    
    
    
    # ## LSTM
    
    # # References
    # [https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html](https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html)
    
    # In[ ]:




