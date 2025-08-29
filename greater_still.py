import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from torch.optim.lr_scheduler import OneCycleLR
import os
import argparse
import math

os.makedirs('models', exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Sudoku solver neural network')
    parser.add_argument('--load-model', type=str, help='Path to a saved model checkpoint to continue training from')
    parser.add_argument('--max-samples', type=int, default=500000, help='Maximum number of samples to use for training')
    return parser.parse_args()

class SpatialConvBlock(nn.Module):
    """Convolutional block that preserves spatial structure"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(0.1)
        
        # Skip connection
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.gelu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        
        return F.gelu(out + identity)

class SudokuTransformerBlock(nn.Module):
    """Transformer-like attention for Sudoku constraints"""
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # MLP
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        
        return x

class HybridSudokuNet(nn.Module):
    """Hybrid CNN-Transformer architecture for Sudoku solving"""
    def __init__(self, dropout_rate=0.15):
        super().__init__()
        
        # Input embedding with multiple channels for different features
        self.input_embed = nn.Sequential(
            nn.Conv2d(10, 64, 3, padding=1),  # 10 channels: 9 digits + 1 mask
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Dropout2d(0.05)
        )
        
        # Spatial feature extraction
        self.spatial_blocks = nn.ModuleList([
            SpatialConvBlock(64, 128),
            SpatialConvBlock(128, 256),
            SpatialConvBlock(256, 512),
            SpatialConvBlock(512, 512),
        ])
        
        # Position-aware pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        
        # Transformer for constraint reasoning
        embed_dim = 512
        self.pos_embed = nn.Parameter(torch.randn(81, embed_dim) * 0.02)
        self.transformer_blocks = nn.ModuleList([
            SudokuTransformerBlock(embed_dim, num_heads=8, dropout=dropout_rate)
            for _ in range(4)
        ])
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 9)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _create_input_features(self, x):
        """Create multi-channel input representation"""
        batch_size = x.size(0)
        device = x.device
        
        # Reshape to 9x9 grid
        x_grid = x.reshape(batch_size, 9, 9)
        
        # Create one-hot encoding for each digit (1-9)
        features = torch.zeros(batch_size, 10, 9, 9, device=device)
        
        # Mask channel (1 where cell is empty, 0 where filled)
        mask = (x_grid == 0).float()
        features[:, 0] = mask
        
        # One-hot encoding for digits 1-9
        for digit in range(1, 10):
            features[:, digit] = (x_grid == digit).float()
        
        return features
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Create multi-channel input
        x_features = self._create_input_features(x)
        
        # Spatial feature extraction
        x = self.input_embed(x_features)
        
        for block in self.spatial_blocks:
            x = block(x)
        
        # Reshape for transformer: [batch, 81, embed_dim]
        x = x.view(batch_size, x.size(1), -1).transpose(1, 2)  # [batch, 81, 512]
        
        # Add positional embeddings
        x = x + self.pos_embed.unsqueeze(0)
        
        # Transformer blocks for constraint reasoning
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # Output projection
        outputs = self.output_projection(x)  # [batch, 81, 9]
        
        return outputs

class AdaptiveSudokuLoss(nn.Module):
    """Adaptive loss that adjusts constraint weighting during training"""
    def __init__(self, base_constraint_weight=0.5, max_constraint_weight=2.0):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.base_constraint_weight = base_constraint_weight
        self.max_constraint_weight = max_constraint_weight
        self.current_epoch = 0
    
    def set_epoch(self, epoch):
        self.current_epoch = epoch
    
    def forward(self, outputs, targets):
        batch_size = outputs.size(0)
        
        # Standard cross-entropy loss
        outputs_flat = outputs.reshape(-1, 9)
        targets_flat = targets.reshape(-1)
        ce_loss = self.ce_loss(outputs_flat, targets_flat)
        
        # Dynamic constraint weight (start low, increase over time)
        progress = min(1.0, self.current_epoch / 100)  # Reach max at epoch 100
        constraint_weight = self.base_constraint_weight + progress * (self.max_constraint_weight - self.base_constraint_weight)
        
        # Get probabilities for constraint calculations
        probs = F.softmax(outputs, dim=2)  # [batch, 81, 9]
        grid_probs = probs.reshape(batch_size, 9, 9, 9)  # [batch, row, col, digit]
        
        constraint_loss = 0
        
        # Row constraints
        for digit in range(9):
            row_sums = grid_probs[:, :, :, digit].sum(dim=2)
            constraint_loss += F.mse_loss(row_sums, torch.ones_like(row_sums))
        
        # Column constraints
        for digit in range(9):
            col_sums = grid_probs[:, :, :, digit].sum(dim=1)
            constraint_loss += F.mse_loss(col_sums, torch.ones_like(col_sums))
        
        # 3x3 box constraints
        for digit in range(9):
            box_probs = grid_probs[:, :, :, digit]
            box_sums = torch.zeros(batch_size, 9, device=outputs.device)
            
            for box_idx in range(9):
                box_row = (box_idx // 3) * 3
                box_col = (box_idx % 3) * 3
                box_sum = box_probs[:, box_row:box_row+3, box_col:box_col+3].sum(dim=(1, 2))
                box_sums[:, box_idx] = box_sum
            
            constraint_loss += F.mse_loss(box_sums, torch.ones_like(box_sums))
        
        constraint_loss = constraint_loss / 27
        
        # Confidence loss (encourage confident predictions on filled cells)
        mask = (targets_flat != -1).float()  # Assuming -1 for empty cells, adjust as needed
        confidence_loss = -(probs.reshape(-1, 9) * F.log_softmax(outputs_flat, dim=1)).sum(dim=1)
        confidence_loss = (confidence_loss * mask).sum() / (mask.sum() + 1e-8)
        
        total_loss = ce_loss + constraint_weight * constraint_loss + 0.1 * confidence_loss
        
        return total_loss

def create_enhanced_datasets(df, test_size=0.2, val_size=0.1, max_samples=500000):
    """Enhanced dataset creation with better data augmentation"""
    
    if len(df) > max_samples:
        print(f"Sampling {max_samples} from {len(df)} total samples")
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
    
    def parse_grid(grid):
        grid_str = str(grid).replace('[','').replace(']','').replace(',','').replace(' ','').strip()
        parsed = [int(x) for x in grid_str if x.isdigit()]
        return parsed if len(parsed) == 81 else None
    
    def validate_solution(solution):
        """Validate that a solution follows Sudoku rules"""
        if not all(1 <= x <= 9 for x in solution):
            return False
        
        grid = np.array(solution).reshape(9, 9)
        
        # Check rows
        for row in grid:
            if len(set(row)) != 9:
                return False
        
        # Check columns
        for col in range(9):
            if len(set(grid[:, col])) != 9:
                return False
        
        # Check 3x3 boxes
        for box_row in range(3):
            for box_col in range(3):
                box = grid[box_row*3:(box_row+1)*3, box_col*3:(box_col+1)*3].flatten()
                if len(set(box)) != 9:
                    return False
        
        return True
    
    print("Processing and validating data...")
    valid_inputs = []
    valid_outputs = []
    
    for i in range(len(df)):
        inp = parse_grid(df.iloc[i, 0])
        out = parse_grid(df.iloc[i, 1])
        
        if inp is not None and out is not None and validate_solution(out):
            valid_inputs.append(inp)
            valid_outputs.append(out)
        
        if (i + 1) % 25000 == 0:
            print(f"Processed {i + 1}/{len(df)} samples, valid: {len(valid_inputs)}")
    
    print(f"Valid samples: {len(valid_inputs)}")
    
    # Convert to numpy
    X = np.array(valid_inputs, dtype=np.float32)
    y = np.array(valid_outputs, dtype=np.int64) - 1  # Convert 1-9 to 0-8
    
    # Stratified split based on puzzle difficulty (number of given clues)
    difficulties = np.sum(X > 0, axis=1)  # Count given clues
    difficulty_bins = np.digitize(difficulties, bins=[0, 25, 30, 35, 40, 81])
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=difficulty_bins)
    
    # Adjust validation size
    val_size_adjusted = val_size / (1 - test_size)
    difficulty_bins_temp = difficulty_bins[:(len(difficulties) - len(X_test))]
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=42, 
        stratify=difficulty_bins_temp)
    
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

def train_epoch(model, optimizer, criterion, train_loader, device, epoch):
    model.train()
    criterion.set_epoch(epoch)  # Update adaptive loss
    
    total_loss = 0.0
    correct_cells = 0
    total_cells = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Adaptive gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        
        # Calculate accuracy
        preds = outputs.reshape(-1, 9).argmax(dim=1)
        targets_flat = targets.reshape(-1)
        correct_cells += (preds == targets_flat).sum().item()
        total_cells += targets.numel()
        
        # Print progress every 1000 batches
        if batch_idx % 1000 == 0 and batch_idx > 0:
            current_acc = 100.0 * correct_cells / total_cells
            print(f"  Batch {batch_idx}/{len(train_loader)}: Loss={loss.item():.4f}, Acc={current_acc:.2f}%")
    
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
    
    train_dataset, val_dataset, test_dataset = create_enhanced_datasets(
        df, max_samples=args.max_samples
    )
    
    # Optimized batch size and data loading
    BATCH_SIZE = 64  # Increased batch size for better gradient estimates
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                            num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=4, pin_memory=True, persistent_workers=True)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Initialize model and training components
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = HybridSudokuNet(dropout_rate=0.15).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Optimizer with better hyperparameters
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=2e-4,  # Lower base learning rate for OneCycle
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Adaptive loss function
    criterion = AdaptiveSudokuLoss(
        base_constraint_weight=0.3,
        max_constraint_weight=1.5
    )
    
    # OneCycle learning rate scheduler
    num_epochs = 200
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-3,  # Peak learning rate
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # Warm up for 10% of training
        div_factor=10,  # Initial lr = max_lr / div_factor
        final_div_factor=100  # Final lr = max_lr / final_div_factor
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
    
    # Training loop
    patience_counter = 0
    patience_limit = 30
    min_epochs = 50
    
    print("Starting training...")
    for epoch in range(start_epoch, num_epochs):
        # Training
        train_loss, train_cell_acc = train_epoch(model, optimizer, criterion, train_loader, device, epoch)
        
        # Validation
        val_loss, val_puzzle_acc, val_cell_acc = evaluate(model, val_loader, criterion, device)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{num_epochs} (LR: {current_lr:.6f})")
        print(f"  Train - Loss: {train_loss:.4f}, Cell Acc: {train_cell_acc:.2f}%")
        print(f"  Val   - Loss: {val_loss:.4f}, Puzzle Acc: {val_puzzle_acc:.2f}%, Cell Acc: {val_cell_acc:.2f}%")
        
        # Save best model
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
            }, 'models/best_hybrid_sudoku_model.pt')
            print(f"  → New best model saved! Cell accuracy: {val_cell_acc:.2f}%")
        else:
            patience_counter += 1
        
        # Regular checkpoint
        if (epoch + 1) % 25 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'val_cell_acc': val_cell_acc,
                'val_puzzle_acc': val_puzzle_acc,
            }, f'models/hybrid_checkpoint_epoch_{epoch+1}.pt')
        
        # Early stopping
        if epoch >= min_epochs and patience_counter >= patience_limit:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Step scheduler after each epoch
        if not isinstance(scheduler, OneCycleLR):  # OneCycle is stepped per batch
            scheduler.step()
        
        print()
    
    # Final evaluation
    print("Evaluating on test set...")
    model_path = 'models/best_hybrid_sudoku_model.pt'
    if os.path.exists(model_path):
        print(f"Loading best model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_puzzle_acc, test_cell_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Results - Loss: {test_loss:.4f}, Puzzle Acc: {test_puzzle_acc:.2f}%, Cell Acc: {test_cell_acc:.2f}%")

if __name__ == "__main__":
    main()