import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F

import os, subprocess
import tqdm
import random
import subprocess
import urllib.request
import tarfile
import re

from config import *
from dataset.hme_dataset import HMEDataset

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, bidirectional=True):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, 
                            bidirectional=bidirectional, batch_first=True)
        
    def forward(self, x, lengths):
        # pack the padded sequences for the LSTM
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        o, (h, c) = self.lstm(packed)
        o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True)
        return o, (h, c)
    
class Attention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super(Attention, self).__init__()
        # encoder is bidirectional, so its hidden dimension is doubled
        self.attn = nn.Linear((encoder_hidden_dim * 2) + decoder_hidden_dim, decoder_hidden_dim)
        self.v = nn.Parameter(torch.rand(decoder_hidden_dim))

    def forward(self, hidden, encoder_outputs, mask):
        # hidden: (batch, decoder_hidden_dim)
        # encoder_outputs: (batch, seq_len, encoder_hidden_dim*2)
        batch_size = encoder_outputs.shape[0]
        seq_len = encoder_outputs.shape[1]
    
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # (batch, seq_len, decoder_hidden_dim)
        energy = energy.transpose(1, 2)  # (batch, decoder_hidden_dim, seq_len)
        v = self.v.repeat(batch_size, 1).unsqueeze(1)  # (batch, 1, decoder_hidden_dim)
        attn_weights = torch.bmm(v, energy).squeeze(1)  # (batch, seq_len)
        attn_weights = attn_weights.masked_fill(mask == 0, -1e10)
        return F.softmax(attn_weights, dim=1)
    
class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, encoder_hidden_dim, decoder_hidden_dim, num_layers=1):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, embed_dim)
        # input to the LSTM will be the embedding concatenated with the context vector from attention
        self.lstm = nn.LSTM(embed_dim + encoder_hidden_dim * 2, decoder_hidden_dim, num_layers=num_layers, batch_first=True)
        self.attention = Attention(encoder_hidden_dim, decoder_hidden_dim)
        # output layer takes the LSTM output, context vector, and embedding to predict the next token
        self.fc_out = nn.Linear(decoder_hidden_dim + encoder_hidden_dim * 2 + embed_dim, output_dim)
    
    def forward(self, input, hidden, cell, encoder_outputs, mask):
        # input: (batch,) current token indices
        input = input.unsqueeze(1)  # (batch, 1)
        embedded = self.embedding(input)  # (batch, 1, embed_dim)
        
        # attention weights and context vector from encoder outputs
        attn_weights = self.attention(hidden[-1], encoder_outputs, mask)  # (batch, seq_len)
        attn_weights = attn_weights.unsqueeze(1)  # (batch, 1, seq_len)
        context = torch.bmm(attn_weights, encoder_outputs)  # (batch, 1, encoder_hidden_dim*2)
        
        # embedded input and context vector
        lstm_input = torch.cat((embedded, context), dim=2)  # (batch, 1, embed_dim + encoder_hidden_dim*2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        
        output = output.squeeze(1)
        context = context.squeeze(1)
        embedded = embedded.squeeze(1)
        # next token
        prediction = self.fc_out(torch.cat((output, context, embedded), dim=1))  # (batch, output_dim)
        return prediction, hidden, cell, attn_weights

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def create_mask(self, src):
        # mask to ignore padding (assuming padded values are all zeros)
        # src shape: (batch, seq_len, feature_dim)
        mask = (src.sum(dim=2) != 0)  # (batch, seq_len)
        return mask

    def forward(self, src, src_lengths, trg, teacher_forcing_ratio=0.5):
        # src: (batch, src_seq_len, feature_dim)
        # trg: (batch, trg_seq_len) where each element is a token index
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        encoder_outputs, (hidden, cell) = self.encoder(src, src_lengths)
        
        # if the encoder is bidirectional, combine the two directions for each layer
        if self.encoder.bidirectional:
            # hidden: (num_layers*2, batch, hidden_dim) -> reshape to (num_layers, 2, batch, hidden_dim)
            hidden = hidden.view(self.encoder.num_layers, 2, hidden.size(1), hidden.size(2)).sum(dim=1)
            cell = cell.view(self.encoder.num_layers, 2, cell.size(1), cell.size(2)).sum(dim=1)
        
        # mask for attention
        mask = self.create_mask(src)
        
        # first input to the decoder is the <sos> token (index 0)
        input_token = trg[:, 0]
        
        for t in range(1, trg_len):
            o, h, c, _ = self.decoder(input_token, hidden, cell, encoder_outputs, mask)
            outputs[:, t] = o
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = o.argmax(1)
            input_token = trg[:, t] if teacher_force else top1
        
        return outputs

def create_model():
    input_dim = 11                   
    enc_hidden_dim = 256             
    dec_hidden_dim = 256             
    embed_dim = 128                  
    output_dim = LATEX_VOCAB_SIZE    
    encoder_num_layers = 2
    decoder_num_layers = 2

    encoder = Encoder(input_dim, enc_hidden_dim, num_layers=encoder_num_layers, bidirectional=True)
    decoder = Decoder(output_dim, embed_dim, enc_hidden_dim, dec_hidden_dim, num_layers=decoder_num_layers)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
    return model

def train(model, train_loader, val_loader, epochs, optimizer, criterion):
    if not os.path.exists("model"):
        os.makedirs("model")

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        print(f"starting epoch {epoch}:")
        for idx, batch in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            # Get the inputs (data is a list of [inputs, labels])
            
            # inputs, latex_gt = data     # latex ground truth
            # inputs = inputs.to(DEVICE)
            # latex_gt = latex_gt.to(DEVICE)

            inputs, lengths, targets = batch
            inputs = inputs.to(DEVICE)
            lengths = lengths.to(DEVICE)
            targets = targets.to(DEVICE)
            
            # optimizer.zero_grad()
            # outputs = model(inputs)
            # loss = criterion(outputs, labels)
            # loss.backward()
            # optimizer.step()

            optimizer.zero_grad()
            output = model(inputs, lengths, targets, teacher_forcing_ratio=0.5)

            # loss = loss.detach().cpu().numpy()
            # inputs = inputs.detach().cpu().numpy()
            # labels = labels.detach().cpu().numpy()
            # running_loss += loss

            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            targets = targets[:, 1:].reshape(-1)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        avg_train_loss = running_loss / len(train_loader)
        print(f"epoch {epoch} training loss: {avg_train_loss:.4f}")

        # evaluate the accuracy after each epoch
        # acc = model.evaluate(model, val_loader, classes, device)
        # if acc > best_acc:
        #     print(f"Better validation accuracy achieved: {acc * 100:.2f}%")
        #     best_acc = acc
        #     print(f"Saving this model as: {my_best_model}")
        #     torch.save(model.state_dict(), my_best_model)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, lengths, targets = batch
                inputs = inputs.to(DEVICE)
                lengths = lengths.to(DEVICE)
                targets = targets.to(DEVICE)
                
                output = model(inputs, lengths, targets, teacher_forcing_ratio=0.0)  # no teacher forcing during evaluation
                output_dim = output.shape[-1]
                output = output[:, 1:].reshape(-1, output_dim)
                targets_flat = targets[:, 1:].reshape(-1)
                loss = criterion(output, targets_flat)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"epoch {epoch} validation loss: {avg_val_loss:.4f}")
        
        # save model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"model/model_best_{epoch}.pth")
            print(f"model saved as 'model/model_best_{epoch}.pth'.")

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # create model
    model = create_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=LATEX_PAD_TOKEN)  

    root_dir = download_data()
    print(root_dir)
    # init dataset

    train_dataset   = HMEDataset(root_dir, "train")
    valid_dataset   = HMEDataset(root_dir, "valid")
    test_dataset    = HMEDataset(root_dir, "test")
    
    print(f"Found {len(train_dataset.ink_files)} files in {train_dataset.split} split")
    print(f"Found {len(valid_dataset.ink_files)} files in {valid_dataset.split} split")
    print(f"Found {len(test_dataset.ink_files)} files in {test_dataset.split} split")
    
    print(train_dataset[0])
    

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, collate_fn=collate_variable_length_sequences)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, collate_fn=collate_variable_length_sequences)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, collate_fn=collate_variable_length_sequences)
    
    print(train_dataloader)
    # print(train_dataloader[0])
    
    # inspect one batch
    for batch in train_dataloader:
        features, lengths, labels = batch
        print(lengths)
    
    train(model, train_dataloader, valid_dataloader, EPOCHS, optimizer, criterion)



def download_data(url="https://storage.googleapis.com/mathwriting_data/mathwriting-2024-excerpt.tgz"):
    filename = url.split("/")[-1]
    dirname = filename.split('.')[0]

    # download the file if it doesn't already exist
    # if not os.path.exists(dirname):
    #     subprocess.run(["wget", "-nc", url], check=True)
    #     subprocess.run(["tar", "zxf", filename], check=True)
    # else: print(f"{filename} already exists. Skipping download.")

    # # extract the archive
    # if os.path.exists(filename):os.remove(filename)
    # else:print(f"Tar file {filename} not found for deletion.")

    if not os.path.exists(dirname):
        if not os.path.exists(filename):
            print(f"downloading {filename}...")
            urllib.request.urlretrieve(url, filename)
        else:
            print(f"{filename} already downloaded.")

        print(f"extracting {filename}...")
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall()
        
        os.remove(filename)
        print(f"extraction complete. removed {filename}.")
    else:
        print(f"{dirname} already exists. skipping download.")


    return dirname

def tokenize_latex(latex_str, vocab):
    tokens = [vocab['<sos>']]
    
    pattern = re.compile(r'(\\[a-zA-Z]+)|(\d+)|(\S)')
    for match in pattern.finditer(latex_str):
        token = match.group(0)
        if token in vocab:
            tokens.append(vocab[token])
        else:
            for char in token:
                if char in vocab:
                    tokens.append(vocab[char])
    tokens.append(vocab['<eos>'])

    return tokens

def collate_variable_length_sequences(batch):
    feature_vectors, labels = zip(*batch)
    # print(len(feature_vectors), len(labels))
    
    # padded_features will have shape: [batch_size, max_seq_len, feature_dim]
    padded_features = pad_sequence(feature_vectors, batch_first=True, padding_value=0.0)
    lengths = torch.tensor([vec.size(0) for vec in feature_vectors]) # original lengths
    
    tokenized_labels = [torch.tensor(tokenize_latex(label, LATEX_VOCAB), dtype=torch.long) for label in labels]
    padded_labels = pad_sequence(tokenized_labels, batch_first=True, padding_value=LATEX_VOCAB['<pad>'])

    return padded_features, lengths, padded_labels

if __name__ == "__main__":
    main()