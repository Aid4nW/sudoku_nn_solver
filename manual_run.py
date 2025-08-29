import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np

# Define the same model architecture as in training
class SudokuMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(81, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 81)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def parse_grid(grid_str):
    grid_str = str(grid_str).replace('[','').replace(']','').replace(',',' ').replace('  ',' ').replace('  ',' ').strip()
    if ',' in grid_str:
        return [int(x) for x in grid_str.split(',') if x.strip()]
    else:
        return [int(x) for x in grid_str.split() if x.strip()]

def print_grid(grid):
    grid = np.array(grid).reshape(9, 9)
    for row in grid:
        print(' '.join(str(int(x)) for x in row))

def main():
    if len(sys.argv) < 3:
        print("Usage: python manual_run.py <model_path> <sudoku_grid>")
        print("Example: python manual_run.py models/sudoku_mlp_epoch_5.pt '0,0,3,...,0'")
        sys.exit(1)

    model_path = sys.argv[1]
    grid_input = sys.argv[2]

    # Prepare model
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = SudokuMLP().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Prepare input
    grid = parse_grid(grid_input)
    print(grid)
    if len(grid) != 81:
        print("Input grid must have 81 values.")
        sys.exit(1)
    input_tensor = torch.tensor([grid], dtype=torch.float32).to(device)

    # Run model
    with torch.no_grad():
        output = model(input_tensor)
        solved = torch.round(output).cpu().numpy().flatten()

    print("Solved Sudoku grid:")
    print_grid(solved)

if __name__ == "__main__":
    main()
