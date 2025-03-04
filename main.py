import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import os, subprocess
import tqdm

from config import *
from dataset.hme_dataset import HMEDataset

def create_model():
    pass

# some basic framework, idk yet
def train(model, train_loader, epochs):
    model.train()
    
    
    for epoch in range(epochs):
        print(f"Starting epoch {epoch}:")
        for idx, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            # Get the inputs (data is a list of [inputs, labels])
            
            inputs, latex_gt = data     # latex ground truth
            inputs = inputs.to(DEVICE)
            latex_gt = latex_gt.to(DEVICE)
            
            # optimizer.zero_grad()
            # outputs = model(inputs)
            # loss = criterion(outputs, labels)
            # loss.backward()
            # optimizer.step()

            loss = loss.detach().cpu().numpy()
            inputs = inputs.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            running_loss += loss

        # evaluate the accuracy after each epoch
        # acc = model.evaluate(model, val_loader, classes, device)
        # if acc > best_acc:
        #     print(f"Better validation accuracy achieved: {acc * 100:.2f}%")
        #     best_acc = acc
        #     print(f"Saving this model as: {my_best_model}")
        #     torch.save(model.state_dict(), my_best_model)
    
    pass


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # create model
    model = create_model()
    
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
    
    for batch in train_dataloader:
        features, lengths, labels = batch
        print(lengths)
    
    # train(model, train_dataloader, EPOCHS)



def download_data(url="https://storage.googleapis.com/mathwriting_data/mathwriting-2024-excerpt.tgz"):
    filename = url.split("/")[-1]
    dirname = filename.split('.')[0]

    # download the file if it doesn't already exist
    if not os.path.exists(dirname):
        subprocess.run(["wget", "-nc", url], check=True)
        subprocess.run(["tar", "zxf", filename], check=True)
    else: print(f"{filename} already exists. Skipping download.")

    # extract the archive
    if os.path.exists(filename):os.remove(filename)
    else:print(f"Tar file {filename} not found for deletion.")


    return dirname

def collate_variable_length_sequences(batch):
    feature_vectors, labels = zip(*batch)
    print(len(feature_vectors), len(labels))
    
    # padded_features will have shape: [batch_size, max_seq_len, feature_dim]
    padded_features = pad_sequence(feature_vectors, batch_first=True, padding_value=0.0)
    lengths = torch.tensor([vec.size(0) for vec in feature_vectors]) # original lengths
    
    return padded_features, lengths, list(labels)

if __name__ == "__main__":
    main()