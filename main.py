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

os.makedirs('models', exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Sudoku solver neural network')
    parser.add_argument('--load-model', type=str, help='Path to a saved model checkpoint to continue training from')
    parser.add_argument('--max-samples', type=int, default=1000000, help='Maximum number of samples to use for training')
    parser.add_argument('--use-iterative', action='store_true', help='Use iterative refinement during inference')
    return parser.parse_args()

class EnhancedSpatialConvBlock(nn.Module):
    """Enhanced convolutional block with attention and better regularization"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, use_attention=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 1)  # 1x1 for channel mixing
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.dropout = nn.Dropout2d(0.08)  # Reduced dropout for better performance
        
        # Channel attention mechanism
        self.use_attention = use_attention
        if use_attention:
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_channels, out_channels // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels // 4, out_channels, 1),
                nn.Sigmoid()
            )
        
        # Skip connection
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        # Layer scale for better gradient flow
        self.layer_scale = nn.Parameter(torch.ones(out_channels, 1, 1) * 0.1)
    
    def forward(self, x):
        identity = self.skip(x)
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.gelu(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.gelu(out)
        
        # Third conv block (1x1)
        out = self.conv3(out)
        out = self.bn3(out)
        
        # Channel attention
        if self.use_attention:
            attention = self.channel_attention(out)
            out = out * attention
        
        out = self.dropout(out)
        
        # Residual with layer scaling
        out = out * self.layer_scale + identity
        return F.gelu(out)

class SudokuConstraintAttention(nn.Module):
    """Specialized attention that focuses on Sudoku constraints"""
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Separate attention for different constraint types
        self.row_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.col_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.box_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        # Constraint mixing
        self.constraint_mixer = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Ensure input is contiguous before reshaping operations
        x = x.contiguous()
        
        # Reshape to grid for constraint-specific attention
        grid_x = x.view(batch_size * 9, 9, self.embed_dim)  # For row attention
        
        # Row-wise attention (each row attends to itself)
        row_out, _ = self.row_attention(grid_x, grid_x, grid_x)
        row_out = row_out.contiguous().view(batch_size, 81, self.embed_dim)
        
        # Column-wise attention
        col_x = x.view(batch_size, 9, 9, self.embed_dim).transpose(1, 2).contiguous()
        col_x = col_x.view(batch_size * 9, 9, self.embed_dim)
        col_out, _ = self.col_attention(col_x, col_x, col_x)
        col_out = col_out.view(batch_size, 9, 9, self.embed_dim).transpose(1, 2).contiguous()
        col_out = col_out.view(batch_size, 81, self.embed_dim)
        
        # 3x3 box attention
        box_x = self._reshape_for_boxes(x)  # [batch*9, 9, embed_dim]
        box_out, _ = self.box_attention(box_x, box_x, box_x)
        box_out = self._reshape_from_boxes(box_out, batch_size)
        
        # Combine all constraint types
        combined = torch.cat([row_out, col_out, box_out], dim=-1)
        mixed = self.constraint_mixer(combined)
        
        return self.norm(x + mixed)
    
    def _reshape_for_boxes(self, x):
        batch_size = x.size(0)
        # Ensure input is contiguous before view operation
        x = x.contiguous()
        grid = x.view(batch_size, 9, 9, self.embed_dim)
        boxes = []
        for i in range(3):
            for j in range(3):
                box = grid[:, i*3:(i+1)*3, j*3:(j+1)*3, :].contiguous()
                boxes.append(box.view(batch_size, 9, self.embed_dim))
        stacked = torch.stack(boxes, dim=1).contiguous()
        return stacked.view(batch_size * 9, 9, self.embed_dim)
    
    def _reshape_from_boxes(self, box_out, batch_size):
        # Ensure input is contiguous before view operations
        box_out = box_out.contiguous()
        box_out = box_out.view(batch_size, 9, 9, self.embed_dim)
        grid = torch.zeros(batch_size, 9, 9, self.embed_dim, device=box_out.device)
        box_idx = 0
        for i in range(3):
            for j in range(3):
                box_view = box_out[:, box_idx, :, :].contiguous().view(batch_size, 3, 3, self.embed_dim)
                grid[:, i*3:(i+1)*3, j*3:(j+1)*3, :] = box_view
                box_idx += 1
        return grid.contiguous().view(batch_size, 81, self.embed_dim)

class EnhancedSudokuTransformerBlock(nn.Module):
    """Enhanced transformer block with constraint-specific attention"""
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.constraint_attention = SudokuConstraintAttention(embed_dim, num_heads, dropout)
        self.global_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Layer scaling
        self.layer_scale1 = nn.Parameter(torch.ones(embed_dim) * 0.1)
        self.layer_scale2 = nn.Parameter(torch.ones(embed_dim) * 0.1)
        self.layer_scale3 = nn.Parameter(torch.ones(embed_dim) * 0.1)
    
    def forward(self, x):
        # Constraint-specific attention
        constraint_out = self.constraint_attention(x)
        x = self.norm1(x + self.layer_scale1 * constraint_out)
        
        # Global self-attention
        global_out, _ = self.global_attention(x, x, x)
        x = self.norm2(x + self.layer_scale2 * global_out)
        
        # MLP
        mlp_out = self.mlp(x)
        x = self.norm3(x + self.layer_scale3 * mlp_out)
        
        return x

class IterativeRefinementSudokuNet(nn.Module):
    """Enhanced Sudoku solver with iterative refinement capability"""
    def __init__(self, dropout_rate=0.1, num_refinement_steps=3):
        super().__init__()
        
        self.num_refinement_steps = num_refinement_steps
        
        # Enhanced input embedding
        self.input_embed = nn.Sequential(
            nn.Conv2d(11, 96, 3, padding=1),  # 11 channels: 9 digits + 1 mask + 1 confidence
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.Dropout2d(0.05),
            nn.Conv2d(96, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        
        # Enhanced spatial feature extraction
        self.spatial_blocks = nn.ModuleList([
            EnhancedSpatialConvBlock(128, 192, use_attention=False),
            EnhancedSpatialConvBlock(192, 256, use_attention=True),
            EnhancedSpatialConvBlock(256, 384, use_attention=True),
            EnhancedSpatialConvBlock(384, 512, use_attention=True),
            EnhancedSpatialConvBlock(512, 512, use_attention=True),
        ])
        
        # Enhanced transformer for constraint reasoning
        embed_dim = 512
        self.pos_embed = nn.Parameter(torch.randn(81, embed_dim) * 0.02)
        self.transformer_blocks = nn.ModuleList([
            EnhancedSudokuTransformerBlock(embed_dim, num_heads=8, dropout=dropout_rate)
            for _ in range(6)  # More transformer layers
        ])
        
        # Iterative refinement module
        self.refinement_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim + 9, embed_dim),  # Previous prediction + features
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(embed_dim, embed_dim)
            ) for _ in range(num_refinement_steps)
        ])
        
        # Enhanced output projection
        self.output_projection = nn.Sequential(
            nn.Linear(embed_dim, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(384, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(192, 9)
        )
        
        # Confidence prediction head
        self.confidence_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
            nn.Sigmoid()
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
    
    def _create_enhanced_input_features(self, x, previous_predictions=None, confidences=None):
        """Create enhanced multi-channel input with iterative information"""
        batch_size = x.size(0)
        device = x.device
        
        # Reshape to 9x9 grid
        x_grid = x.reshape(batch_size, 9, 9)
        
        # Create features: 11 channels total
        features = torch.zeros(batch_size, 11, 9, 9, device=device)
        
        # Mask channel (1 where cell is empty, 0 where filled)
        mask = (x_grid == 0).float()
        features[:, 0] = mask
        
        # One-hot encoding for digits 1-9
        for digit in range(1, 10):
            features[:, digit] = (x_grid == digit).float()
        
        # Confidence channel for iterative refinement
        if confidences is not None:
            features[:, 10] = confidences.reshape(batch_size, 9, 9)
        else:
            features[:, 10] = torch.ones_like(mask)
        
        return features
    
    def forward(self, x, use_iterative=False):
        batch_size = x.size(0)
        
        # Initial forward pass
        outputs, confidences = self._single_forward_pass(x)
        
        if not use_iterative or not self.training:
            return outputs
        
        # Iterative refinement during training
        for step in range(self.num_refinement_steps):
            # Create enhanced input with previous predictions
            previous_probs = F.softmax(outputs, dim=-1)
            max_probs, _ = previous_probs.max(dim=-1)
            
            # Enhanced input features
            enhanced_features = self._create_enhanced_input_features(x, previous_probs, max_probs)
            
            # Refine predictions
            refined_features = self._extract_features(enhanced_features)
            refined_features = refined_features.view(batch_size, 81, -1)
            
            # Add positional embeddings
            refined_features = refined_features + self.pos_embed.unsqueeze(0)
            
            # Apply refinement
            prev_pred_features = torch.cat([refined_features, previous_probs], dim=-1)
            refined_features = self.refinement_blocks[step](prev_pred_features)
            
            # Update outputs
            outputs = self.output_projection(refined_features)
            confidences = self.confidence_head(refined_features).squeeze(-1)
        
        return outputs
    
    def _single_forward_pass(self, x):
        batch_size = x.size(0)
        
        # Create multi-channel input
        x_features = self._create_enhanced_input_features(x)
        
        # Extract features
        features = self._extract_features(x_features)
        
        # Reshape for transformer
        features = features.view(batch_size, features.size(1), -1).transpose(1, 2)
        
        # Add positional embeddings
        features = features + self.pos_embed.unsqueeze(0)
        
        # Transformer processing
        for transformer_block in self.transformer_blocks:
            features = transformer_block(features)
        
        # Output predictions and confidence
        outputs = self.output_projection(features)
        confidences = self.confidence_head(features).squeeze(-1)
        
        return outputs, confidences
    
    def _extract_features(self, x_features):
        # Ensure input is contiguous and properly shaped
        x_features = x_features.contiguous()
        
        # Spatial feature extraction
        x = self.input_embed(x_features)
        
        for block in self.spatial_blocks:
            x = block(x)
        
        return x
    
    def predict_with_iterative_refinement(self, x, num_iterations=5):
        """Inference with iterative refinement for maximum accuracy"""
        self.eval()
        with torch.no_grad():
            batch_size = x.size(0)
            
            # Initial prediction
            outputs, confidences = self._single_forward_pass(x)
            
            # Iterative refinement
            for iteration in range(num_iterations):
                # Get current predictions
                probs = F.softmax(outputs, dim=-1)
                max_probs, predictions = probs.max(dim=-1)
                
                # Create mask for low-confidence predictions
                confidence_threshold = 0.9
                uncertain_mask = max_probs < confidence_threshold
                
                # If all predictions are confident, stop early
                if not uncertain_mask.any():
                    break
                
                # Refine uncertain predictions
                enhanced_features = self._create_enhanced_input_features(x, probs, max_probs)
                refined_features = self._extract_features(enhanced_features)
                refined_features = refined_features.view(batch_size, 81, -1)
                refined_features = refined_features + self.pos_embed.unsqueeze(0)
                
                # Apply transformer refinement
                for transformer_block in self.transformer_blocks:
                    refined_features = transformer_block(refined_features)
                
                # Update outputs
                new_outputs = self.output_projection(refined_features)
                new_confidences = self.confidence_head(refined_features).squeeze(-1)
                
                # Blend predictions based on confidence
                blend_factor = uncertain_mask.float().unsqueeze(-1)
                outputs = outputs * (1 - blend_factor) + new_outputs * blend_factor
                confidences = confidences * (1 - uncertain_mask.float()) + new_confidences * uncertain_mask.float()
            
            return outputs

class FocalConstraintLoss(nn.Module):
    """Enhanced loss with focal loss and stronger constraint enforcement"""
    def __init__(self, alpha=0.25, gamma=2.0, constraint_weight=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.constraint_weight = constraint_weight
        self.current_epoch = 0
    
    def set_epoch(self, epoch):
        self.current_epoch = epoch
    
    def forward(self, outputs, targets, confidences=None):
        batch_size = outputs.size(0)
        # Ensure input tensors are properly shaped and contiguous
        if len(outputs.shape) != 3:  # Should be [batch_size, 81, 9]
            outputs = outputs.reshape(batch_size, 81, 9)
        if len(targets.shape) != 2:  # Should be [batch_size, 81]
            targets = targets.reshape(batch_size, 81)
        
        # Focal loss for hard examples
        # Ensure proper reshaping with contiguous tensors
        outputs = outputs.contiguous()
        outputs_flat = outputs.reshape(-1, 9)
        targets_flat = targets.reshape(-1)
        
        ce_loss = F.cross_entropy(outputs_flat, targets_flat, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        focal_loss = focal_loss.mean()
        
        # Enhanced constraint loss
        probs = F.softmax(outputs, dim=-1)  # Use last dimension for softmax
        grid_probs = probs.reshape(batch_size, 9, 9, 9).contiguous()
        
        constraint_loss = 0
        
        # Row constraints with higher penalty for violations
        for digit in range(9):
            row_sums = grid_probs[:, :, :, digit].sum(dim=2)
            constraint_loss += F.smooth_l1_loss(row_sums, torch.ones_like(row_sums))
        
        # Column constraints
        for digit in range(9):
            col_sums = grid_probs[:, :, :, digit].sum(dim=1)
            constraint_loss += F.smooth_l1_loss(col_sums, torch.ones_like(col_sums))
        
        # 3x3 box constraints
        for digit in range(9):
            box_probs = grid_probs[:, :, :, digit]
            box_sums = torch.zeros(batch_size, 9, device=outputs.device)
            
            for box_idx in range(9):
                box_row = (box_idx // 3) * 3
                box_col = (box_idx % 3) * 3
                box_sum = box_probs[:, box_row:box_row+3, box_col:box_col+3].sum(dim=(1, 2))
                box_sums[:, box_idx] = box_sum
            
            constraint_loss += F.smooth_l1_loss(box_sums, torch.ones_like(box_sums))
        
        constraint_loss = constraint_loss / 27
        
        # Confidence regularization
        confidence_loss = 0
        if confidences is not None:
            # Ensure confidences are properly shaped
            confidences = confidences.contiguous()
            # Encourage high confidence on correct predictions
            correct_mask = (outputs_flat.argmax(dim=1) == targets_flat).float()
            confidence_loss = F.binary_cross_entropy(
                confidences.reshape(-1), correct_mask, reduction='mean'
            )
        
        # Dynamic constraint weighting
        epoch_factor = min(1.0, self.current_epoch / 50)
        dynamic_constraint_weight = self.constraint_weight * (0.5 + 0.5 * epoch_factor)
        
        total_loss = focal_loss + dynamic_constraint_weight * constraint_loss
        if confidences is not None:
            total_loss += 0.1 * confidence_loss
        
        return total_loss

# [Rest of the training code remains largely the same, with these key changes:]

def train_epoch(model, optimizer, criterion, train_loader, device, epoch):
    model.train()
    criterion.set_epoch(epoch)
    
    total_loss = 0.0
    correct_cells = 0
    total_cells = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # Use iterative refinement during training every 3rd epoch
        use_iterative = (epoch + 1) % 3 == 0
        outputs = model(inputs, use_iterative=use_iterative)
        
        # Get confidences if available
        confidences = None
        if hasattr(model, '_last_confidences'):
            confidences = model._last_confidences
        
        loss = criterion(outputs, targets, confidences)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        
        # Calculate accuracy
        preds = outputs.reshape(-1, 9).argmax(dim=1)
        targets_flat = targets.reshape(-1)
        correct_cells += (preds == targets_flat).sum().item()
        total_cells += targets.numel()
        
        # Progress reporting
        if batch_idx % 1000 == 0 and batch_idx > 0:
            current_acc = 100.0 * correct_cells / total_cells
            print(f"  Batch {batch_idx}/{len(train_loader)}: Loss={loss.item():.4f}, Acc={current_acc:.2f}%")
    
    avg_loss = total_loss / len(train_loader.dataset)
    cell_accuracy = 100.0 * correct_cells / total_cells
    
    return avg_loss, cell_accuracy

def evaluate_with_iterative(model, data_loader, criterion, device, use_iterative=False):
    model.eval()
    total_loss = 0.0
    correct_puzzles = 0
    correct_cells = 0
    total_cells = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            if use_iterative and hasattr(model, 'predict_with_iterative_refinement'):
                outputs = model.predict_with_iterative_refinement(inputs, num_iterations=5)
            else:
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

# Additional helper function for data preprocessing
def create_enhanced_datasets(df, test_size=0.2, val_size=0.1, max_samples=1000000):
    """Enhanced dataset creation with better validation and stratification"""
    
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
        
        # Check rows, columns, and boxes
        for i in range(9):
            if len(set(grid[i, :])) != 9 or len(set(grid[:, i])) != 9:
                return False
        
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
        
        if (i + 1) % 50000 == 0:
            print(f"Processed {i + 1}/{len(df)} samples, valid: {len(valid_inputs)}")
    
    print(f"Valid samples: {len(valid_inputs)}")
    
    # Convert to numpy
    X = np.array(valid_inputs, dtype=np.float32)
    y = np.array(valid_outputs, dtype=np.int64) - 1  # Convert 1-9 to 0-8
    
    # Stratified split
    difficulties = np.sum(X > 0, axis=1)
    difficulty_bins = np.digitize(difficulties, bins=[0, 25, 30, 35, 40, 81])
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=difficulty_bins)
    
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

def main():
    args = parse_args()
    
    # Load data
    print("Loading data...")
    df = pd.read_csv('sudoku.csv')
    
    train_dataset, val_dataset, test_dataset = create_enhanced_datasets(
        df, max_samples=args.max_samples
    )
    
    # Data loaders with optimized settings
    BATCH_SIZE = 48  # Slightly smaller due to larger model
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                            num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=4, pin_memory=True, persistent_workers=True)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Initialize enhanced model
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = IterativeRefinementSudokuNet(dropout_rate=0.1, num_refinement_steps=2).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Enhanced optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=1e-4,  # Lower learning rate for stability
        weight_decay=0.02,  # Slightly higher weight decay
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Focal constraint loss
    criterion = FocalConstraintLoss(
        alpha=0.25,
        gamma=2.0,
        constraint_weight=2.5
    )
    
    # Conservative learning rate schedule
    num_epochs = 150
    scheduler = OneCycleLR(
        optimizer,
        max_lr=3e-4,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.05,
        div_factor=20,
        final_div_factor=200
    )
    
    # Training loop
    start_epoch = 0
    best_val_cell_acc = 0
    patience_counter = 0
    patience_limit = 25
    
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
    
    print("Starting enhanced training...")
    for epoch in range(start_epoch, num_epochs):
        # Training
        train_loss, train_cell_acc = train_epoch(model, optimizer, criterion, train_loader, device, epoch)
        
        # Standard validation
        val_loss, val_puzzle_acc, val_cell_acc = evaluate_with_iterative(
            model, val_loader, criterion, device, use_iterative=False)
        
        # Iterative validation every 5 epochs
        if (epoch + 1) % 5 == 0:
            iter_val_loss, iter_val_puzzle_acc, iter_val_cell_acc = evaluate_with_iterative(
                model, val_loader, criterion, device, use_iterative=True)
            print(f"  Iterative Val - Loss: {iter_val_loss:.4f}, Puzzle Acc: {iter_val_puzzle_acc:.2f}%, Cell Acc: {iter_val_cell_acc:.2f}%")
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{num_epochs} (LR: {current_lr:.6f})")
        print(f"  Train - Loss: {train_loss:.4f}, Cell Acc: {train_cell_acc:.2f}%")
        print(f"  Val   - Loss: {val_loss:.4f}, Puzzle Acc: {val_puzzle_acc:.2f}%, Cell Acc: {val_cell_acc:.2f}%")
        
        # Save best model
        if val_cell_acc > best_val_cell_acc:
            best_val_cell_acc = val_cell_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'val_cell_acc': val_cell_acc,
                'val_puzzle_acc': val_puzzle_acc,
            }, 'models/best_enhanced_sudoku_model.pt')
            print(f"  → New best model saved! Cell accuracy: {val_cell_acc:.2f}%")
        
        # Regular checkpoint
        if (epoch + 1) % 20 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'val_cell_acc': val_cell_acc,
                'val_puzzle_acc': val_puzzle_acc,
            }, f'models/enhanced_checkpoint_epoch_{epoch+1}.pt')
        
        print()
    
    # Final evaluation with iterative refinement
    print("Final evaluation with iterative refinement...")
    model_path = 'models/best_enhanced_sudoku_model.pt'
    if os.path.exists(model_path):
        print(f"Loading best model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Standard test
    test_loss, test_puzzle_acc, test_cell_acc = evaluate_with_iterative(model, test_loader, criterion, device, use_iterative=False)
    print(f"Standard Test Results - Loss: {test_loss:.4f}, Puzzle Acc: {test_puzzle_acc:.2f}%, Cell Acc: {test_cell_acc:.2f}%")
    
    # Iterative test
    iter_test_loss, iter_test_puzzle_acc, iter_test_cell_acc = evaluate_with_iterative(model, test_loader, criterion, device, use_iterative=True)
    print(f"Iterative Test Results - Loss: {iter_test_loss:.4f}, Puzzle Acc: {iter_test_puzzle_acc:.2f}%, Cell Acc: {iter_test_cell_acc:.2f}%")

if __name__ == "__main__":
    main()