"""
Part C: Sequence Modeling with RNN for POI Prediction

This script trains a 'SimpleRNN' to predict POI probabilities from user movement sequences generated from PartB. 
Key features include:
- Loading and preprocessing sequence data from multiple cities.
- Mapping coordinates to embeddings and POI probabilities. 
- Custom Dataset and DataLoader for variable-length sequences.
- Training, validation, and loss viz. 

Outputs:
- Trained RNN model (`rnn_model.pth`).
- Timestamped loss plot for performance tracking
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from numpy import ceil
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

torch.manual_seed(8)  # Set the seed for CPU

# Set device if possible!
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load phrase embeddings
phrase_embeddings = torch.load('../phrase_embeddings.pt').to(device)
# print(f"Phrase embeddings shape: {phrase_embeddings.shape}")  # Should be (85, 768)

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, lengths):
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(device)
        # Pack the padded sequence
        x_packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        # Pass through RNN
        out_packed, _ = self.rnn(x_packed, h0)
        # Unpack the sequence
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        # Gather the outputs at the last valid time step
        idx = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, out.size(2)).to(device)
        out = out.gather(1, idx).squeeze(1)
        # Pass through fully connected layer
        out = self.fc(out)
        # Apply log softmax
        out = nn.functional.log_softmax(out, dim=1)
        return out

def city_to_dataset_path(city):
    base_path = '../Part C preprocessed data'
    return os.path.join(base_path, f'city{city}_sequences' + ('.parquet' if city == 'A' else '.txt'))

def city_to_mapping_path(city):
    base_path = '../Part C preprocessed data'
    return os.path.join(base_path, f'city{city}_mapping.csv')

def txt_to_sequences(txt_file_path):
    sequences = []
    with open(txt_file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # Extract the sequence
            sequence = line.split('[')[1].split(']')[0]
            sequence = sequence.split('),')
            sequence = [tuple(map(lambda y: ceil(int(y)/20), x.replace('(', '').replace(')', '').split(','))) for x in sequence]
            sequences.append(sequence)
    return sequences

def parquet_to_sequences(parquet_file_path):
    sequences_df = pd.read_parquet(parquet_file_path)
    sequences = []
    for seq_str in sequences_df['sequence_str']:
        # Split the string by ';' to get individual coordinates
        str_split = seq_str.split(';')
        # Split each coordinate and convert to integers
        sequence = [tuple(map(lambda y: int(ceil(int(y)/10000)), x.split(','))) for x in str_split]
        sequences.append(sequence)
    return sequences

def city_to_sequences(city):
    if city == 'A':
        return parquet_to_sequences(city_to_dataset_path(city))
    else:
        return txt_to_sequences(city_to_dataset_path(city))

def coordinates_to_vector_map(path_to_mapping_csv):
    # Load the mapping
    df = pd.read_csv(path_to_mapping_csv)
    # Load POI probabilities and input features
    df["POI_Probabilities"] = df.iloc[:, 2:(2+85)].apply(lambda x: torch.tensor(x.values, dtype=torch.float32).to(device), axis=1)
    df['Input_features'] = df.iloc[:, (2+85):(2+85+768)].apply(lambda x: torch.tensor(x.values, dtype=torch.float32).to(device), axis=1)
    # Keep only necessary columns
    df = df[['x', 'y', 'POI_Probabilities', 'Input_features']]
    return df

def sequence_to_vectors(tup_sequence, coordinates_to_vector_map_df, min_length=2):
    if len(tup_sequence) < min_length:
        raise ValueError(f'The sequence must contain at least {min_length} coordinates')
    
    df = coordinates_to_vector_map_df
    input_vector_sequence = []
    target_labels = None

    for i, tup in enumerate(tup_sequence):
        x, y = tup
        series = df.loc[(df['x'] == x) & (df['y'] == y)]
        
        if series.empty:
            # Handle missing coordinates by using a uniform distribution
            poi_probabilities = torch.ones(85, device=device) / 85  # Shape: (85,)
            # Corrected matrix multiplication
            input_feature = torch.matmul(phrase_embeddings.T, poi_probabilities)  # Shape: (768,)
        else:
            poi_probabilities = series['POI_Probabilities'].values[0]
            # Normalize POI probabilities
            if poi_probabilities.sum() != 0:
                poi_probabilities = poi_probabilities / poi_probabilities.sum()
            else:
                poi_probabilities = torch.ones(85, device=device) / 85  # Uniform distribution
            input_feature = series['Input_features'].values[0]
        
        if i == len(tup_sequence) - 1:
            # Last coordinate's POI probabilities are the target
            target_labels = poi_probabilities
        else:
            # Previous coordinates are used as input features
            input_vector_sequence.append(input_feature)
    
    if not input_vector_sequence:
        raise ValueError('The input vector sequence cannot be empty')
    return torch.stack(input_vector_sequence), target_labels

def sequence_to_vectors_from_city(tup_sequence, city):
    mapping = coordinates_to_vector_map(city_to_mapping_path(city))
    return sequence_to_vectors(tup_sequence, mapping, min_length=1)

class SequenceDataset(Dataset):
    def __init__(self, city_names):
        self.sequences = []
        self.mappings = []
        # Ensure city_names is a list
        if not isinstance(city_names, list):
            city_names = [city_names]
        for city in city_names:
            seqs = city_to_sequences(city)
            mapping = coordinates_to_vector_map(city_to_mapping_path(city))
            for seq in seqs:
                if len(seq) < 2:
                    continue
                self.sequences.append(seq)
                self.mappings.append(mapping)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        input_vector_sequence, target_labels = sequence_to_vectors(self.sequences[idx], self.mappings[idx])
        return input_vector_sequence, target_labels

def collate_fn(batch):
    sequences_unpadded, targets = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences_unpadded], dtype=torch.long)
    sequences_padded = nn.utils.rnn.pad_sequence(sequences_unpadded, batch_first=True)
    targets = torch.stack(targets)
    return sequences_padded.to(device), targets.to(device), lengths.to(device)

# Create DataLoaders
train_dataset = SequenceDataset(['A', 'B'])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

validation_dataset = SequenceDataset('C')
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

test_dataset = SequenceDataset('D')
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Hyperparameters
input_size = 768
hidden_size = 128
output_size = 85
num_layers = 2
learning_rate = 0.0005
num_epochs = 30

# Initialize model, loss function, and optimizer
model = SimpleRNN(input_size, hidden_size, output_size, num_layers).to(device)
criterion = nn.KLDivLoss(reduction='batchmean')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if __name__ == '__main__':
    print("Training the model...")

    # Training loop
    train_losses = []
    validation_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for sequences, targets, lengths in train_loader:
            optimizer.zero_grad()
            outputs = model(sequences, lengths)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        average_train_loss = total_train_loss / len(train_loader)
        train_losses.append(average_train_loss)
        
        # Validation
        model.eval()
        total_validation_loss = 0
        with torch.no_grad():
            for sequences, targets, lengths in validation_loader:
                outputs = model(sequences, lengths)
                loss = criterion(outputs, targets)
                total_validation_loss += loss.item()
        average_validation_loss = total_validation_loss / len(validation_loader)
        validation_losses.append(average_validation_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {average_train_loss:.4f}, Validation Loss: {average_validation_loss:.4f}')

    # Save the model
    torch.save(model.state_dict(), 'rnn_model.pth')
    print("Model saved to 'rnn_model.pth'")

    # Generate a timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # Plot the losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), validation_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

    # Save the plot with a timestamp
    filename = f'loss_plot_{timestamp}.png'
    plt.savefig(filename, format='png', dpi=300)
    print(f"Plot saved as '{filename}'") 

    # Compute the testing loss
    model.eval()  # Set the model to evaluation mode
    total_test_loss = 0
    with torch.no_grad():  # Disable gradient computation
        for sequences, targets, lengths in test_loader:
            sequences = sequences.to(torch.float32)
            lengths = lengths.to(torch.long)
            targets = targets.to(torch.float32)
            
            outputs = model(sequences, lengths)
            loss = criterion(outputs, targets)
            total_test_loss += loss.item()

    average_test_loss = total_test_loss / len(test_loader)
    print(f'Test Loss: {average_test_loss:.4f}')

    # End of script