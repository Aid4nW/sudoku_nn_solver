import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import argparse

os.makedirs('models', exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Sudoku solver neural network')
    parser.add_argument('--load-model', type=str, help='Path to a saved model checkpoint to continue training from')
    parser.add_argument('--max-samples', type=int, default=50000, help='Maximum number of samples to use for training')
    return parser.parse_args()

class MemoryEfficientSudokuMLP(nn.Module):
    """Enhanced model with wider layers, skip connections, and normalization"""
    def __init__(self, dropout_rate=0.3):  # Increased dropout
        super(MemoryEfficientSudokuMLP, self).__init__()
        
        # Use simpler encoding: just normalize 0-9 to 0-1
        input_size = 81  # Raw input, normalized
        hidden_size = 1024  # Increased width
        
        # Input processing
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)  # Added layer norm
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Deep network with consistent width
        self.hidden1 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        self.hidden3 = nn.Linear(hidden_size, hidden_size)
        self.bn4 = nn.BatchNorm1d(hidden_size)
        self.ln4 = nn.LayerNorm(hidden_size)
        self.dropout4 = nn.Dropout(dropout_rate)
        
        # Output processing
        self.pre_output = nn.Linear(hidden_size, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.ln5 = nn.LayerNorm(512)
        self.dropout5 = nn.Dropout(dropout_rate/2)  # Less dropout near output
        
        # Output layer
        self.output_layer = nn.Linear(512, 81 * 9)
        
    def forward(self, x):
        # Normalize input (0-9 -> 0-1)
        x = x / 9.0
        
        # Input processing with dual normalization
        x = self.input_layer(x)
        x = self.bn1(x)
        x = self.ln1(x)
        x = F.gelu(x)  # GELU activation
        x = self.dropout1(x)
        
        # First hidden layer with enhanced residual
        residual = x
        x = self.hidden1(x)
        x = self.bn2(x)
        x = self.ln2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = x + residual
        
        # Second hidden layer with enhanced residual
        residual = x
        x = self.hidden2(x)
        x = self.bn3(x)
        x = self.ln3(x)
        x = F.gelu(x)
        x = self.dropout3(x)
        x = x + residual
        
        # Third hidden layer with enhanced residual
        residual = x
        x = self.hidden3(x)
        x = self.bn4(x)
        x = self.ln4(x)
        x = F.gelu(x)
        x = self.dropout4(x)
        x = x + residual
        
        # Pre-output processing
        x = self.pre_output(x)
        x = self.bn5(x)
        x = self.ln5(x)
        x = F.gelu(x)
        x = self.dropout5(x)
        
        # Output layer
        x = self.output_layer(x)
        x = x.view(-1, 81, 9)
        
        return x

def encode_sudoku_grid_simple(grid):
    """Simple encoding - just return the grid as float array"""
    return np.array(grid, dtype=np.float32)

def create_memory_efficient_datasets(df, test_size=0.2, val_size=0.1, max_samples=50000):
    """Memory efficient version using simple encoding"""
    
    # Sample subset for memory efficiency
    if len(df) > max_samples:
        print(f"Sampling {max_samples} from {len(df)} total samples")
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
    
    def parse_grid(grid):
        grid_str = str(grid).replace('[','').replace(']','').replace(',','').replace(' ','').strip()
        parsed = [int(x) for x in grid_str if x.isdigit()]
        return parsed if len(parsed) == 81 else None
    
    print("Processing data...")
    valid_inputs = []
    valid_outputs = []
    
    for i in range(len(df)):
        inp = parse_grid(df.iloc[i, 0])
        out = parse_grid(df.iloc[i, 1])
        
        if inp is not None and out is not None:
            valid_inputs.append(inp)
            valid_outputs.append(out)
        
        if (i + 1) % 10000 == 0:
            print(f"Processed {i + 1}/{len(df)} samples, valid: {len(valid_inputs)}")
    
    print(f"Valid samples: {len(valid_inputs)}")
    
    # Convert to numpy with memory-efficient dtypes
    X = np.array(valid_inputs, dtype=np.float32)
    y = np.array(valid_outputs, dtype=np.int64) - 1
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=None)
    
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=None)
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )
    
    return train_dataset, val_dataset, test_dataset

class SudokuConstraintLoss(nn.Module):
    """Custom loss that combines CrossEntropy with constraint penalties"""
    def __init__(self, constraint_weight=0.1):
        super(SudokuConstraintLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.constraint_weight = constraint_weight
    
    def forward(self, outputs, targets):
        # Standard cross-entropy loss on flattened outputs
        batch_size = outputs.size(0)
        outputs_flat = outputs.reshape(-1, 9)  # Combine batch and cells dimensions
        targets_flat = targets.reshape(-1)     # Flatten targets
        ce_loss = self.ce_loss(outputs_flat, targets_flat)
        
        # Constraint penalties (soft constraints)
        constraint_loss = 0
        
        # Apply softmax to get probabilities
        probs = F.softmax(outputs, dim=2)  # Shape: [batch_size, 81, 9]
        
        # Reshape to [batch_size, 9, 9, 9] to work with rows and columns
        probs = probs.reshape(batch_size, 9, 9, 9)
        
        # Row constraints
        row_sums = probs.sum(dim=2)  # Sum along row dimension
        constraint_loss += F.mse_loss(row_sums, torch.ones_like(row_sums))
        
        # Column constraints
        col_sums = probs.sum(dim=1)  # Sum along column dimension
        constraint_loss += F.mse_loss(col_sums, torch.ones_like(col_sums))
        
        # Add box constraints (3x3 boxes)
        box_sums = probs.reshape(batch_size, 9, 3, 3, 9).sum(dim=(2, 3))
        constraint_loss += F.mse_loss(box_sums, torch.ones_like(box_sums))
        
        # Average the constraint losses
        constraint_loss = constraint_loss / 3  # Average of row, column, and box constraints
        
        return ce_loss + self.constraint_weight * constraint_loss

def create_improved_datasets(df, test_size=0.2, val_size=0.1, max_samples=50000):
    """Create train/val/test splits with proper data encoding - memory efficient version"""
    
    # First, sample a smaller subset to avoid memory issues
    if len(df) > max_samples:
        print(f"Sampling {max_samples} from {len(df)} total samples for memory efficiency")
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
    
    # Parse grids
    def parse_grid(grid):
        grid_str = str(grid).replace('[','').replace(']','').replace(',','').replace(' ','').strip()
        parsed = [int(x) for x in grid_str if x.isdigit()]
        return parsed if len(parsed) == 81 else None
    
    print("Parsing grids...")
    inputs_parsed = df.iloc[:, 0].apply(parse_grid)
    outputs_parsed = df.iloc[:, 1].apply(parse_grid)
    
    # Filter valid grids and encode in batches to save memory
    print("Encoding grids...")
    valid_inputs = []
    valid_outputs = []
    
    batch_size = 1000
    total_processed = 0
    
    for i in range(0, len(df), batch_size):
        batch_end = min(i + batch_size, len(df))
        
        for j in range(i, batch_end):
            inp = inputs_parsed.iloc[j]
            out = outputs_parsed.iloc[j]
            
            if inp is not None and out is not None and len(inp) == 81 and len(out) == 81:
                valid_inputs.append(encode_sudoku_grid_simple(inp))
                valid_outputs.append(out)
        
        total_processed = batch_end
        if total_processed % 5000 == 0:
            print(f"Processed {total_processed}/{len(df)} samples, valid: {len(valid_inputs)}")
    
    print(f"Valid samples: {len(valid_inputs)} out of {len(df)}")
    
    if len(valid_inputs) == 0:
        raise ValueError("No valid samples found!")
    
    # Convert to numpy arrays in smaller chunks to avoid memory spike
    print("Converting to numpy arrays...")
    X = np.array(valid_inputs, dtype=np.float32)  # Use float32 to save memory
    y = np.array(valid_outputs, dtype=np.int64) - 1  # Convert 1-9 to 0-8
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)
    
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=42)
    
    # Convert to tensors
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )
    
    return train_dataset, val_dataset, test_dataset

def train_epoch(model, optimizer, criterion, train_loader, device):
    model.train()
    total_loss = 0.0
    correct_cells = 0
    total_cells = 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        # Pass outputs directly to criterion which handles reshaping
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        
        # Calculate cell-wise accuracy using flattened predictions
        preds = outputs.reshape(-1, 9).argmax(dim=1)
        targets_flat = targets.reshape(-1)
        correct_cells += (preds == targets_flat).sum().item()
        total_cells += targets.numel()
    
    avg_loss = total_loss / len(train_loader.dataset)
    cell_accuracy = 100.0 * correct_cells / total_cells
    
    return avg_loss, cell_accuracy

def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct_puzzles = 0
    correct_cells = 0
    total_cells = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            # Pass outputs directly to criterion which handles reshaping
            loss = criterion(outputs, targets)
            
            total_loss += loss.item() * inputs.size(0)
            
            # Get predictions using reshaped outputs
            batch_size = outputs.size(0)
            preds = outputs.reshape(batch_size, 81, 9).argmax(dim=2)
            
            # Puzzle-wise accuracy (all 81 cells correct)
            correct_puzzles += (preds == targets).all(dim=1).sum().item()
            
            # Cell-wise accuracy (using flattened predictions)
            preds_flat = outputs.reshape(-1, 9).argmax(dim=1)
            targets_flat = targets.reshape(-1)
            correct_cells += (preds_flat == targets_flat).sum().item()
            total_cells += targets.numel()
    
    avg_loss = total_loss / len(data_loader.dataset)
    puzzle_accuracy = 100.0 * correct_puzzles / len(data_loader.dataset)
    cell_accuracy = 100.0 * correct_cells / total_cells
    
    return avg_loss, puzzle_accuracy, cell_accuracy

def main():
    args = parse_args()
    
    # Load and prepare data
    print("Loading data...")
    df = pd.read_csv('sudoku.csv')
    
    # Use memory-efficient dataset creation
    train_dataset, val_dataset, test_dataset = create_memory_efficient_datasets(
        df, max_samples=args.max_samples
    )
    
    # Create data loaders with smaller batch size
    BATCH_SIZE = 16  # Even smaller for memory efficiency
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Initialize model and training components
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Use the memory-efficient model
    model = MemoryEfficientSudokuMLP(dropout_rate=0.2).to(device)
    
    # Adjusted learning rate and weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01, betas=(0.9, 0.999))
    
    # Use constraint loss with higher weight to enforce Sudoku rules more strongly
    criterion = SudokuConstraintLoss(constraint_weight=0.5)  # Increased from 0.2 to 0.5
    
    # More aggressive learning rate scheduling
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max',
        factor=0.3,  # More aggressive reduction
        patience=3,   # Less patience before reducing
        min_lr=1e-6,
    )
    
    # Load checkpoint if specified
    start_epoch = 0
    best_val_cell_acc = 0
    if args.load_model and os.path.exists(args.load_model):
        print(f"Loading checkpoint from {args.load_model}")
        checkpoint = torch.load(args.load_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Start from next epoch
        best_val_cell_acc = checkpoint.get('val_cell_acc', 0)
        print(f"Resuming from epoch {start_epoch}, best validation cell accuracy: {best_val_cell_acc:.2f}%")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Training loop with adjusted parameters
    num_epochs = 100  # More epochs for better convergence
    patience_counter = 0
    patience_limit = 20  # More patience for finding global optimum
    min_epochs = 25    # Ensure minimum training time
    total_epochs = start_epoch + num_epochs
    
    print("Starting training...")
    for epoch in range(start_epoch, total_epochs):
        # Training
        train_loss, train_cell_acc = train_epoch(model, optimizer, criterion, train_loader, device)
        
        # Validation
        val_loss, val_puzzle_acc, val_cell_acc = evaluate(model, val_loader, criterion, device)
        
        # Learning rate scheduling based on cell accuracy
        scheduler.step(val_cell_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train - Loss: {train_loss:.4f}, Cell Acc: {train_cell_acc:.2f}%")
        print(f"  Val   - Loss: {val_loss:.4f}, Puzzle Acc: {val_puzzle_acc:.2f}%, Cell Acc: {val_cell_acc:.2f}%")
        
        # Save best model based on cell accuracy
        if val_cell_acc > best_val_cell_acc:
            best_val_cell_acc = val_cell_acc
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_cell_acc': val_cell_acc,
                'val_puzzle_acc': val_puzzle_acc,
            }, 'models/best_sudoku_model.pt')
            print(f"  → New best model saved! Cell accuracy: {val_cell_acc:.2f}%")
        else:
            patience_counter += 1
        
        # Early stopping with minimum epochs requirement
        if epoch >= min_epochs and patience_counter >= patience_limit:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        print()
    
    # Final evaluation on test set
    print("Evaluating on test set...")
    model_path = 'models/best_sudoku_model.pt'
    if os.path.exists(model_path):
        print(f"Loading best model from {model_path}")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("No saved model found, using current model state")
    
    test_loss, test_puzzle_acc, test_cell_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Results - Loss: {test_loss:.4f}, Puzzle Acc: {test_puzzle_acc:.2f}%, Cell Acc: {test_cell_acc:.2f}%")

if __name__ == "__main__":
    main()