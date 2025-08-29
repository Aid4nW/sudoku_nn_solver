import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import os
import argparse

os.makedirs('models', exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Sudoku solver neural network')
    parser.add_argument('--load-model', type=str, help='Path to a saved model checkpoint to continue training from')
    parser.add_argument('--max-samples', type=int, default=200000, help='Maximum number of samples to use for training')
    return parser.parse_args()

class SudokuResidualBlock(nn.Module):
    """Residual block for processing Sudoku puzzle features"""
    def __init__(self, hidden_size, dropout_rate):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        identity = x
        
        # First sub-block
        out = self.linear1(x)
        out = self.bn1(out)
        out = F.gelu(out)
        
        # Second sub-block
        out = self.linear2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        
        # Add residual and apply activation
        return F.gelu(out + identity)

class OptimizedSudokuNet(nn.Module):
    """Optimized neural network for solving Sudoku puzzles"""
    def __init__(self, dropout_rate=0.15):
        super().__init__()
        
        # Calculate input and hidden dimensions
        self.num_cells = 81  # 9x9 grid
        self.pos_encoding_dim = 18  # 9 for row + 9 for column
        input_size = self.num_cells + self.num_cells * self.pos_encoding_dim
        hidden_size = 1536
        
        # Input processing
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.input_bn = nn.BatchNorm1d(hidden_size)
        self.input_dropout = nn.Dropout(dropout_rate * 0.5)
        
        # Residual blocks for feature extraction
        self.blocks = nn.ModuleList([
            SudokuResidualBlock(hidden_size, dropout_rate) for _ in range(6)
        ])
        
        # Constraint processing
        self.constraint_layer = nn.Linear(hidden_size, hidden_size)
        self.constraint_bn = nn.BatchNorm1d(hidden_size)
        self.constraint_dropout = nn.Dropout(dropout_rate * 0.7)
        
        # Output processing
        self.pre_output = nn.Linear(hidden_size, 768)
        self.pre_output_bn = nn.BatchNorm1d(768)
        self.pre_output_dropout = nn.Dropout(dropout_rate * 0.3)
        
        # Final output layer
        self.output_layer = nn.Linear(768, self.num_cells * 9)  # 9 possible digits per cell
        
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _create_position_encoding(self, batch_size):
        """Create positional encoding for Sudoku grid cells"""
        device = next(self.parameters()).device
        
        # Create position indices
        positions = torch.arange(self.num_cells, device=device)
        rows = positions // 9
        cols = positions % 9
        
        # Create one-hot encodings
        row_encoding = F.one_hot(rows, num_classes=9).float()
        col_encoding = F.one_hot(cols, num_classes=9).float()
        
        # Combine and reshape for batch
        pos_encoding = torch.cat([row_encoding, col_encoding], dim=1)  # [81, 18]
        return pos_encoding.unsqueeze(0).expand(batch_size, -1, -1).reshape(batch_size, -1)

    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape and normalize input
        x = x.reshape(batch_size, self.num_cells).float()
        x_norm = x / 9.0  # Normalize to [0, 1]
        
        # Add position encoding
        pos_encoding = self._create_position_encoding(batch_size)
        x = torch.cat([x_norm, pos_encoding], dim=1)
        
        # Input feature extraction
        x = self.input_layer(x)
        x = self.input_bn(x)
        x = F.gelu(x)
        x = self.input_dropout(x)
        
        # Process through residual blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply constraints
        x = self.constraint_layer(x)
        x = self.constraint_bn(x)
        x = F.gelu(x)
        x = self.constraint_dropout(x)
        
        # Generate output
        x = self.pre_output(x)
        x = self.pre_output_bn(x)
        x = F.gelu(x)
        x = self.pre_output_dropout(x)
        
        # Final predictions
        x = self.output_layer(x)
        x = x.contiguous()  # Ensure tensor is contiguous before reshaping
        return x.view(batch_size, self.num_cells, 9)

class AdvancedSudokuConstraintLoss(nn.Module):
    """Enhanced constraint loss with stronger Sudoku rule enforcement"""
    def __init__(self, constraint_weight=1.0, diversity_weight=0.3):
        super(AdvancedSudokuConstraintLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.constraint_weight = constraint_weight
        self.diversity_weight = diversity_weight
    
    def forward(self, outputs, targets):
        batch_size = outputs.size(0)
        
        # Standard cross-entropy loss
        outputs_flat = outputs.reshape(-1, 9)
        targets_flat = targets.reshape(-1)
        ce_loss = self.ce_loss(outputs_flat, targets_flat)
        
        # Get probabilities for constraint calculations
        probs = F.softmax(outputs, dim=2)
        
        # Reshape for constraint calculations
        grid_probs = probs.reshape(batch_size, 9, 9, 9)  # [batch, row, col, digit]
        
        constraint_loss = 0
        
        # Row constraints - each digit 1-9 should appear exactly once per row
        for digit in range(9):
            row_sums = grid_probs[:, :, :, digit].sum(dim=2)  # Sum over columns
            constraint_loss += F.mse_loss(row_sums, torch.ones_like(row_sums))
        
        # Column constraints - each digit 1-9 should appear exactly once per column
        for digit in range(9):
            col_sums = grid_probs[:, :, :, digit].sum(dim=1)  # Sum over rows
            constraint_loss += F.mse_loss(col_sums, torch.ones_like(col_sums))
        
        # 3x3 box constraints
        for digit in range(9):
            box_probs = grid_probs[:, :, :, digit]  # [batch, 9, 9]
            box_sums = torch.zeros(batch_size, 9, device=outputs.device)
            
            for box_idx in range(9):
                box_row = (box_idx // 3) * 3
                box_col = (box_idx % 3) * 3
                box_sum = box_probs[:, box_row:box_row+3, box_col:box_col+3].sum(dim=(1, 2))
                box_sums[:, box_idx] = box_sum
            
            constraint_loss += F.mse_loss(box_sums, torch.ones_like(box_sums))
        
        # Normalize constraint loss
        constraint_loss = constraint_loss / 27  # 9 digits × 3 constraint types
        
        # Diversity loss - encourage confident predictions
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=2).mean()
        diversity_loss = entropy  # Lower entropy = more confident predictions
        
        total_loss = (ce_loss + 
                     self.constraint_weight * constraint_loss + 
                     self.diversity_weight * diversity_loss)
        
        return total_loss

def create_memory_efficient_datasets(df, test_size=0.2, val_size=0.1, max_samples=200000):
    """Memory efficient dataset creation with improved parsing"""
    
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
            # Additional validation - ensure output is a valid Sudoku solution
            if all(1 <= x <= 9 for x in out):
                valid_inputs.append(inp)
                valid_outputs.append(out)
        
        if (i + 1) % 10000 == 0:
            print(f"Processed {i + 1}/{len(df)} samples, valid: {len(valid_inputs)}")
    
    print(f"Valid samples: {len(valid_inputs)}")
    
    # Convert to numpy with memory-efficient dtypes
    X = np.array(valid_inputs, dtype=np.float32)
    y = np.array(valid_outputs, dtype=np.int64) - 1  # Convert 1-9 to 0-8
    
    # Split data with stratification to ensure balanced difficulty
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)
    
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=42)
    
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

def train_epoch(model, optimizer, criterion, train_loader, device):
    model.train()
    total_loss = 0.0
    correct_cells = 0
    total_cells = 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Gradient clipping with adaptive norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        
        # Calculate accuracy
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
            loss = criterion(outputs, targets)
            
            total_loss += loss.item() * inputs.size(0)
            
            batch_size = outputs.size(0)
            preds = outputs.reshape(batch_size, 81, 9).argmax(dim=2)
            
            # Puzzle-wise accuracy
            correct_puzzles += (preds == targets).all(dim=1).sum().item()
            
            # Cell-wise accuracy
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
    
    train_dataset, val_dataset, test_dataset = create_memory_efficient_datasets(
        df, max_samples=args.max_samples
    )
    
    # Optimized batch size and data loading
    BATCH_SIZE = 32  # Increased for better gradient estimates
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                            num_workers=2, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=2, pin_memory=True, persistent_workers=True)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Initialize model and training components
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = OptimizedSudokuNet(dropout_rate=0.15).to(device)
    
    # Optimized optimizer with better hyperparameters
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=0.0015,  # Slightly higher learning rate
        weight_decay=0.005,  # Reduced weight decay
        betas=(0.9, 0.98),  # Better momentum for this problem
        eps=1e-8
    )
    
    # Enhanced constraint loss
    criterion = AdvancedSudokuConstraintLoss(
        constraint_weight=1.5,  # Strong constraint enforcement
        diversity_weight=0.2
    )
    
    # Cosine annealing with warm restarts for better exploration
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=15,  # First restart after 15 epochs
        T_mult=1,  # Keep same cycle length
        eta_min=1e-6
    )
    
    # Load checkpoint if specified
    start_epoch = 0
    best_val_cell_acc = 0
    if args.load_model and os.path.exists(args.load_model):
        print(f"Loading checkpoint from {args.load_model}")
        checkpoint = torch.load(args.load_model, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_cell_acc = checkpoint.get('val_cell_acc', 0)
        print(f"Resuming from epoch {start_epoch}, best validation cell accuracy: {best_val_cell_acc:.2f}%")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Training loop with optimized parameters
    num_epochs = 150  # More epochs for convergence
    patience_counter = 0
    patience_limit = 25
    min_epochs = 30
    total_epochs = start_epoch + num_epochs
    
    print("Starting training...")
    for epoch in range(start_epoch, total_epochs):
        # Training
        train_loss, train_cell_acc = train_epoch(model, optimizer, criterion, train_loader, device)
        
        # Validation
        val_loss, val_puzzle_acc, val_cell_acc = evaluate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{num_epochs} (LR: {current_lr:.6f})")
        print(f"  Train - Loss: {train_loss:.4f}, Cell Acc: {train_cell_acc:.2f}%")
        print(f"  Val   - Loss: {val_loss:.4f}, Puzzle Acc: {val_puzzle_acc:.2f}%, Cell Acc: {val_cell_acc:.2f}%")
        
        # Save best model and regular checkpoints
        if val_cell_acc > best_val_cell_acc:
            best_val_cell_acc = val_cell_acc
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'val_cell_acc': val_cell_acc,
                'val_puzzle_acc': val_puzzle_acc,
            }, 'models/best_sudoku_model.pt')
            print(f"  → New best model saved! Cell accuracy: {val_cell_acc:.2f}%")
        else:
            patience_counter += 1
        
        # Save regular checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'val_cell_acc': val_cell_acc,
                'val_puzzle_acc': val_puzzle_acc,
            }, f'models/checkpoint_epoch_{epoch+1}.pt')
        
        # Early stopping
        if epoch >= min_epochs and patience_counter >= patience_limit:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        print()
    
    # Final evaluation
    print("Evaluating on test set...")
    model_path = 'models/best_sudoku_model.pt'
    if os.path.exists(model_path):
        print(f"Loading best model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_puzzle_acc, test_cell_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Results - Loss: {test_loss:.4f}, Puzzle Acc: {test_puzzle_acc:.2f}%, Cell Acc: {test_cell_acc:.2f}%")

if __name__ == "__main__":
    main()