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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_

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
    
    # ---- YOUR DEVICE ----
    device = get_gpu()
    
    # write to csv to avoid wait time with tokenization
    # clean_data.to_csv('data/fake_news/clean_news_data.csv', index=False)
    dypes = {'text': str, 'label': 'Int64'}
    clean_data = pd.read_csv("data/fake_news/clean_news_data.csv", dtype=dypes)
    # Drop empty strings
    clean_data = clean_data.dropna()
    
    
    """
    ---- CHANGE SAMPLE SIZE IF NEEDED ----
    """
    clean_data = clean_data.sample(n=1000, random_state=0)
    
    # train test split
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
    vocab_dict = build_vocab_from_iterator(iterator=yield_tokens(clean_data), specials=["<unk>"], max_tokens=10)
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
    
    # In[367]:
    
    
    class RNN(nn.Module):
        def __init__(self, input_size, hidden_size):
            super(RNN, self).__init__()
            self.hidden_size = hidden_size
        # this converts your sequence of words to a vector to pass through the NN
            self.word_sequence_to_embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=8)
            self.rnn = nn.RNN(input_size=8, hidden_size=hidden_size, num_layers=4, batch_first=True,  dropout=0.5)
            # self.hidden_layer_input = nn.Linear(in_features=10, out_features=hidden_size, bias=False)
            # self.hidden_layer_previous = nn.Linear(in_f|eatures=hidden_size, out_features=hidden_size, bias=False)
    
            self.hidden_layer_out = nn.Linear(in_features=hidden_size, out_features=1)
            self.activation = nn.Sigmoid()
    
        def forward(self, input_, hidden):
            sequence_embeddings = self.word_sequence_to_embedding(input_)
            # Check if the embedding layer's weights require gradients
            
            # print(sequence_embeddings.shape)
            # hidden = F.tanh(self.hidden_layer_input(sequence_embeddings) + self.hidden_layer_previous(hidden))
            output, hidden = self.rnn(sequence_embeddings, hidden)
            # print(hidden.shape)
            output = self.hidden_layer_out(output)
          
            output = output[:, -1, :]
            output = self.activation(output)
           
            return output, hidden.detach()
    
        def initHidden(self):
        # Return a matrix of 1 row and k columns where k=hidden_size
            return torch.zeros(4, 16, self.hidden_size).to(device)
            
    
    
    # In[368]:
    
    
    
    # Initialize Model
    n_hidden = 256
    input_size = len(vocab_dict)
    rnn_model = RNN(input_size=input_size, hidden_size=n_hidden)
    rnn_model.to(device)
    
    
    # ### Training
    
    # In[377]:
    
    
    def get_labels(results):
        decision = lambda val: 1 if val >= 0.5 else 0
        labels = torch.where(results > 0.5, 1.0, 0.0)
        return labels
    
    
    # In[378]:
    
    
    train_loader = torch.utils.data.DataLoader(train_list_of_tuples, batch_size=16, shuffle=True, collate_fn=collate_batch)
    test_loader = torch.utils.data.DataLoader(test_list_of_tuples, batch_size=16, shuffle=True, collate_fn=collate_batch)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(rnn_model.parameters(), lr=0.05, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2)
    
    
    # In[387]:
    
    
    epoch_iter = 15
    def fit_rnn(model, dataloader, nbr_of_epochs, optimizer, criterion):
        epoch_results = {}
        for epoch in range(nbr_of_epochs):
            # ---- MODEL TRAINING ----
        # Put model in training mode
            hidden = model.initHidden()
        
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
            
            for labels, inputs in dataloader:
            # get the inputs; data is a list of [inputs, labels]
                inputs, labels = inputs.to(device), labels.float().to(device)
    
                hidden = hidden.detach()
                
            # zero the parameter gradients
                optimizer.zero_grad()
                
                
        # ---- FORWARD, BACKWARD, OPTIMIZE ----
            # Get model training predictions
                outputs, hidden = model(inputs, hidden)
                outputs = torch.squeeze(outputs, 1)
                
                
            # Convert model training predictions to their respective classifications
                predicted_labels = get_labels(outputs)
                
                
            # Compute Loss of current batch
                # print(f'Outputs shape: {outputs.shape}, Labels shape: {labels.shape}')
                # print(f'Outputs dtype: {outputs.dtype}, Labels dtype: {labels.dtype}')
                
                loss = criterion(outputs, labels)
                
            # remove outputs
    
            # Compute total number of correctly classified examples
                nbr_of_correct_predictions = torch.sum(predicted_labels == labels).item()
            # Grab batch size
                total_nbr_of_elements = labels.shape[0]
            # Compute number of correctly labeled examples and the total exampes
                training_total_correct += nbr_of_correct_predictions
                # print(total_nbr_of_elements)
                training_total_examples += total_nbr_of_elements
                
            # Backward pass
                loss.backward()
                
    
                # clip_gradients(grad_clip_val=5, model=model)
     
            # Update model params with gradient clippings
                # print(torch.linalg.vector_norm(model.parameters(), 'fro'))
                nn.utils.clip_grad_norm_(model.parameters(), 1.25)
                optimizer.step()
                 
            # update training loss of epoch
                training_losses += float(loss.item())*total_nbr_of_elements
                
            #  update the current batch number of epoch
                batch_nbr += 1
    
            
            # End training time
            end_time = time.perf_counter()
        # Get total runntime of epoch
            epoch_runtime = timedelta(seconds=end_time-start_time).total_seconds()
    
      
        # Compute testing accuracy of epoch
            total_training_epoch_loss = round(training_losses/len(dataloader), 4)
            training_accuracy = round(training_total_correct/training_total_examples, 4)
            scheduler.step(total_training_epoch_loss/len(dataloader))
    
            result_dict = {"training_loss": total_training_epoch_loss, 'training_accuracy': training_accuracy,  "runtime": epoch_runtime}
            print("\n")
            print(f'Epoch {epoch + 1}/{epoch_iter} <-> Runtime: {round(epoch_runtime, 0)}s <-> Training loss: {total_training_epoch_loss} <-> Training Accuracy: {training_accuracy}')
    
    
    # In[ ]:
    
    
    fit_rnn(model=rnn_model, dataloader=train_loader, nbr_of_epochs=epoch_iter, optimizer=optimizer, criterion=criterion)
    
    
    # ### Testing
    
    # In[37]:
    
    
    # for this, that in train_loader:
    #     print(type(this))
    #     break
    
    
    # ## LSTM
    
    # # References
    # [https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html](https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html)
    # 
    # [https://www.analyticsvidhya.com/blog/2020/08/top-4-sentence-embedding-techniques-using-python/](https://www.analyticsvidhya.com/blog/2020/08/top-4-sentence-embedding-techniques-using-python/)
    # 
    # [https://www.geeksforgeeks.org/adjusting-learning-rate-of-a-neural-network-in-pytorch/#](https://www.geeksforgeeks.org/adjusting-learning-rate-of-a-neural-network-in-pytorch/#)
    
    # In[ ]:
    
    
    
    
