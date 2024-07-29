import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import numpy as np
import pandas as pd
from string import punctuation
from collections import Counter
import csv


train_fpath = 'news_data.csv'
train_raw_data = pd.read_csv(train_fpath, encoding='utf-8', header=0, keep_default_na=False)

subset = train_raw_data.sample(n=10000, random_state=1)  # random_state for reproducibility
train_raw_data = subset

texts = [''.join([c for c in text.lower() if c not in punctuation]) for text in train_raw_data['text']]

# split by new lines and spaces
all_text = ' '.join(texts)

# create a list of words
words = all_text.split()

# Load your data
train_fpath = 'news_data.csv'
train_raw_data = pd.read_csv(train_fpath, encoding='utf-8', header=0, keep_default_na=False)

# Tokenize the texts
def tokenize(text):
    return text.split()  # Simple tokenization by splitting on whitespace

# Combine all texts and get word frequencies
all_words = []
for text in train_raw_data['text']:
    all_words.extend(tokenize(text))

# Build a frequency dictionary
counts = Counter(all_words)

# Limit vocabulary size (e.g., 10,000)
vocab_size = 10000
vocab = sorted(counts, key=counts.get, reverse=True)[:vocab_size]
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

# Add a special token for unknown words
vocab_to_int['<UNK>'] = len(vocab_to_int) + 1

# Tokenize each article using the reduced vocabulary
text_ints = []
for text in train_raw_data['text']:
    tokens = tokenize(text)
    text_ints.append([vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in tokens])

# Get the labels (0 and 1) from the dataset
encoded_labels = train_raw_data['label'].tolist()

# Print sample results
print("Sample tokenized text:", text_ints[0])
print("Sample label:", encoded_labels[0])
print("Vocabulary size:", len(vocab_to_int))

# stats about vocabulary
print('Unique words: ', len((vocab_to_int)))

# print tokens in first article
print('Tokenized text: \n', text_ints[:1])

# outlier article stats
text_lens = Counter([len(x) for x in text_ints])
print("Zero-length text: {}".format(text_lens[0]))
print("Maximum text length: {}".format(max(text_lens)))

print('Number of texts before removing outliers: ', len(text_ints))

## remove any articles/labels with zero length from the text_ints list.

# get indices of any articles with length 0
non_zero_idx = [ii for ii, text in enumerate(text_ints) if len(text) != 0]

# remove 0-length articles and their labels
text_ints = [text_ints[ii] for ii in non_zero_idx]
encoded_labels = np.array([encoded_labels[ii] for ii in non_zero_idx])

print('Number of texts after removing outliers: ', len(text_ints))

def pad_features(text_ints, seq_length):
    ''' Return features of text_ints, where each article is padded with 0's 
        or truncated to the input seq_length.
    '''
    
    # getting the correct rows x cols shape
    features = np.zeros((len(text_ints), seq_length), dtype=int)

    # for each article, grab that article and 
    for i, row in enumerate(text_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]
    
    return features

    seq_length = 200

features = pad_features(text_ints, seq_length=seq_length)

# test statements - do not change -
assert len(features)==len(text_ints), "Your features should have as many rows as articles."
assert len(features[0])==seq_length, "Each feature row should contain seq_length values."

# print first 10 values of the first 30 batches 
print(features[:30, :10])

train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Calculate the indices for splitting
total_size = len(features)
train_size = int(total_size * train_ratio)
val_size = int(total_size * val_ratio)

# Split the data
train_x, remaining_x = features[:train_size], features[train_size:]
train_y, remaining_y = encoded_labels[:train_size], encoded_labels[train_size:]

val_x, test_x = remaining_x[:val_size], remaining_x[val_size:]
val_y, test_y = remaining_y[:val_size], remaining_y[val_size:]

# Print out the shapes of your resultant feature data
print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape))
print("Validation set: \t{}".format(val_x.shape))
print("Test set: \t\t{}".format(test_x.shape))

# create Tensor datasets
train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

# dataloaders
batch_size = 10

# make sure the SHUFFLE your training data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=5)
valid_loader = DataLoader(valid_data, shuffle=False, batch_size=batch_size, num_workers=5)
test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size, num_workers=5)

# obtain one batch of training data
dataiter = iter(train_loader)
sample_x, sample_y = next(dataiter)

print('Sample input size: ', sample_x.size())  # batch_size, seq_length
print('Sample input: \n', sample_x)
print()
print('Sample label size: ', sample_y.size())  # batch_size
print('Sample label: \n', sample_y)

# First checking if GPU is available
train_on_gpu = torch.cuda.is_available()

if train_on_gpu:
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')

import torch.nn as nn


class SentimentRNN(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super(SentimentRNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=drop_prob, batch_first=True)

        # dropout layer
        self.dropout = nn.Dropout(0.3)

        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()


    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)

        # embeddings and lstm_out
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)

        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        # sigmoid function
        sig_out = self.sig(out)

        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]  # get last batch of labels

        # return last sigmoid output and hidden state
        return sig_out, hidden


    def init_hidden(self, batch_size):
        """ Initializes hidden state """
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if train_on_gpu:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden

    # Instantiate the model w/ hyperparams
vocab_size = len(vocab_to_int) + 1 # +1 for the 0 padding + our word tokens
output_size = 1
embedding_dim = 200
hidden_dim = 256
n_layers = 3

net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
# move model to GPU, if available
if train_on_gpu:
    net.cuda()
    
print(net)

# loss and optimization functions
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=1, verbose=True)

print("Train dataset size:", len(train_loader.dataset))
print("Validation dataset size:", len(valid_loader.dataset))

# Training parameters
epochs = 1
clip = 5  # Gradient clipping
min_loss = np.inf

# Train for some number of epochs
for e in range(epochs):
    train_loss = 0.0
    valid_loss = 0.0
    num_correct = 0
    total_samples = 0

    net.train()
    for inputs, labels in train_loader:
        batch_size = inputs.size(0)
        if train_on_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()

        h = net.init_hidden(batch_size)
        h = tuple([each.data for each in h])

        net.zero_grad()
        output, h = net(inputs, h)
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        train_loss += loss.item() * batch_size
        total_samples += batch_size

    train_loss /= total_samples

    net.eval()
    num_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            batch_size = inputs.size(0)
            if train_on_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()

            val_h = net.init_hidden(batch_size)
            val_h = tuple([each.data for each in val_h])

            output, val_h = net(inputs, val_h)
            loss = criterion(output.squeeze(), labels.float())

            pred = torch.round(output.squeeze())
            correct_tensor = pred.eq(labels.float().view_as(pred))
            correct = correct_tensor.cpu().numpy() if train_on_gpu else correct_tensor.numpy()
            num_correct += np.sum(correct)
            total_samples += batch_size
            valid_loss += loss.item() * batch_size

    valid_loss /= total_samples
    scheduler.step(valid_loss)

    print(f"Epoch: {e + 1}/{epochs}...",
          f"Train Loss: {train_loss:.6f}...",
          f"Validation Loss: {valid_loss:.6f}",
          f"Accuracy: {num_correct / total_samples:.6f}")

    if min_loss >= valid_loss:
        torch.save(net.state_dict(), 'checkpointx.pth')
        min_loss = valid_loss
        print("Loss decreased. Saving model...")

# Get test data loss and accuracy
test_losses = [] # track loss
num_correct = 0

# init hidden state
h = net.init_hidden(batch_size)

net.eval()
loader = test_loader
# iterate over test data
for inputs, labels in loader:

    # Creating new variables for the hidden state, otherwise
    # we'd backprop through the entire training history
    h = net.init_hidden(inputs.size(0))

    if train_on_gpu:
        inputs, labels = inputs.cuda(), labels.cuda()
    
    # get predicted outputs
    output, h = net(inputs, h)
    
    # calculate loss
    test_loss = criterion(output.squeeze(), labels.float())
    test_losses.append(test_loss.item())
    
    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze())  # rounds to the nearest integer
    
    # compare predictions to true label
    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)


# -- stats! -- ##
# avg test loss
print("Test loss: {:.3f}".format(np.mean(test_losses)))

# accuracy over all test data
test_acc = num_correct/len(loader.dataset)
print("Test accuracy: {:.4f}%".format(test_acc*100))