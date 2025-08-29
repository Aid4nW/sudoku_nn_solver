
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

import os
os.makedirs('models', exist_ok=True)

# Define a simple MLP model for Sudoku
class SudokuMLP(nn.Module):
    def __init__(self):
        super(SudokuMLP, self).__init__()
        self.input_layer = nn.Linear(81, 128)
        self.hidden_layers = nn.ModuleList([nn.Linear(128, 128) for _ in range(13)])
        self.output_layer = nn.Linear(128, 81 * 9)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.output_layer(x)
        x = x.view(-1, 81, 9)  # batch_size x 81 x 9
        return x

# Load the CSV file and randomly sample 40% of the rows to reduce memory usage
df = pd.read_csv('sudoku.csv')
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Assume first column is input, second column is output (both as strings or lists)
inputs = df.iloc[:, 0]
outputs = df.iloc[:, 1]

# Convert string/list representations to lists of ints
def parse_grid(grid):
    # Remove brackets, split by comma, strip whitespace, and convert to int
    grid_str = str(grid).replace('[','').replace(']','').replace(',','').replace(' ','').strip()
    return [int(x) for x in grid_str if x.isdigit()]

inputs_parsed = inputs.apply(parse_grid)
outputs_parsed = outputs.apply(parse_grid)

# Convert to numpy arrays
inputs_np = inputs_parsed.apply(lambda x: x if len(x)==81 else [0]*81).to_list()
outputs_np = outputs_parsed.apply(lambda x: x if len(x)==81 else [0]*81).to_list()

# Split into train/test sets
train_inputs_np, test_inputs_np, train_outputs_np, test_outputs_np = train_test_split(
    inputs_np, outputs_np, test_size=0.3, random_state=42, shuffle=True)



# Convert to PyTorch tensors
train_inputs = torch.tensor(train_inputs_np, dtype=torch.float32)
train_outputs = torch.tensor(train_outputs_np, dtype=torch.long) - 1  # targets: 0-8 for digits 1-9
test_inputs = torch.tensor(test_inputs_np, dtype=torch.float32)
test_outputs = torch.tensor(test_outputs_np, dtype=torch.long) - 1

# Create DataLoaders for mini-batch training
BATCH_SIZE = 64
train_dataset = TensorDataset(train_inputs, train_outputs)
test_dataset = TensorDataset(test_inputs, test_outputs)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train inputs shape: {train_inputs.shape}")
print(f"Train outputs shape: {train_outputs.shape}")
print(f"Test inputs shape: {test_inputs.shape}")
print(f"Test outputs shape: {test_outputs.shape}")

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using {device} device")

# Training and testing functions
def train(model, optimizer, criterion, train_loader, device):
    model.train()
    total_loss = 0.0
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)  # outputs: (batch, 81, 9)
        loss = criterion(outputs.view(-1, 9), targets.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(train_loader.dataset)

def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)  # (batch, 81, 9)
            preds = outputs.argmax(dim=2)  # (batch, 81)
            correct += (preds == targets).all(dim=1).sum().item()
            total += targets.size(0)
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return accuracy

# Training loop
num_epochs = 20
learning_rate = 0.01
model = SudokuMLP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()



for epoch in range(num_epochs):
    loss = train(model, optimizer, criterion, train_loader, device)
    test_acc = test(model, test_loader, device)
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss:.4f} - Test Accuracy: {test_acc:.2f}%")
    # Save the model after each epoch, versioned by epoch number
    model_path = f"models/sudoku_mlp_epoch_{epoch+1}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    # Debug: print a few predictions and their targets from the test set
    model.eval()
    with torch.no_grad():
        for batch_inputs, batch_targets in test_loader:
            sample_inputs = batch_inputs[:3].to(device)
            sample_targets = batch_targets[:3].to(device)
            sample_outputs = model(sample_inputs)
            sample_probs = torch.softmax(sample_outputs, dim=2)
            sample_preds = sample_probs.argmax(dim=2)
            for i in range(min(3, sample_inputs.size(0))):
                print(f"Sample {i+1} prediction: {sample_preds[i].cpu().numpy() + 1}")
                print(f"Sample {i+1} target:     {sample_targets[i].cpu().numpy() + 1}")
                print(f"Sample {i+1} probabilities (first cell): {sample_probs[i,0].cpu().numpy()}")
            break

