# test.py - Run this to load the best DNN model and predict on test.csv
# This script loads a pre-trained PyTorch model and generates predictions on the test dataset.
# It handles CPU-only environments by mapping the model from CUDA to CPU and adapts to pre-processed numerical data.

import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
import torch  # For loading and running the model
import torch.nn as nn  # For defining the neural network architecture
import joblib  # For loading the saved scaler

# Device configuration: Use CPU by default, fallback to GPU if available
device = torch.device("cpu")  # Explicitly set to CPU for consistency
print(f"Using device: {device}")  # Debug: Confirm the device being used

# DNN class definition
class DNN(nn.Module):
    def __init__(self, input_size, hidden_layers=[]):
        """
        Initialize the DNN model with configurable hidden layers.
        
        Args:
            input_size (int): Number of input features (default assumed 32).
            hidden_layers (list): List of integers specifying the size of each hidden layer.
        """
        super(DNN, self).__init__()
        layers = []
        in_size = input_size
        for h in hidden_layers:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            in_size = h
        layers.append(nn.Linear(in_size, 1))  # Output layer for regression
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Model predictions.
        """
        return self.model(x)

# Function to parse binnedInc column to midpoint (only if strings are present)
def parse_bin(s):
    """
    Convert binned income strings to numerical midpoints.
    
    Args:
        s (str): String representing an income bin (e.g., '[10000, 20000)').
    
    Returns:
        float: Midpoint of the bin, or NaN if invalid.
    """
    if isinstance(s, str):  # Check if the input is a string
        s = s.replace('[', '').replace(']', '').replace('(', '').replace(')', '')
        parts = s.split(',')
        if len(parts) == 2:
            low = float(parts[0].strip())
            high = float(parts[1].strip())
            return (low + high) / 2
    return s  # Return the original value (e.g., float) if not a string

# Test function to load model and generate predictions
def test_model(model_path, test_csv_path, hidden_layers, input_size=32):
    """
    Load a pre-trained model and predict on the test dataset.
    
    Args:
        model_path (str): Path to the saved model weights (e.g., 'best_dnn.pth').
        test_csv_path (str): Path to the test CSV file.
        hidden_layers (list): Architecture of the hidden layers.
        input_size (int, optional): Number of input features. Defaults to 32.
    
    Returns:
        numpy.ndarray: Array of predicted values.
    """
    # Initialize the model with the specified architecture
    model = DNN(input_size, hidden_layers)
    
    # Load the model state dict, explicitly mapping to CPU to handle GPU-saved models
    try:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        print(f"Model loaded successfully from {model_path} on {device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    model.to(device)
    model.eval()
    
    # Load and preprocess the test data
    test_df = pd.read_csv(test_csv_path, encoding='latin-1')
    print(f"Initial number of columns in test_df: {test_df.shape[1]}")  # Debug: Check total columns
    print(f"Initial columns: {test_df.columns.tolist()}")  # Debug: List all columns
    
    # Ensure 'Geography' and 'TARGET_deathRate' are dropped to match training data
    columns_to_drop = [col for col in ['Geography', 'TARGET_deathRate'] if col in test_df.columns]
    if columns_to_drop:
        test_df = test_df.drop(columns_to_drop, axis=1)
        print(f"Dropped columns: {columns_to_drop}")
    
    print(f"Number of columns after dropping: {test_df.shape[1]}")  # Debug: Check after drop
    print(f"Final columns: {test_df.columns.tolist()}")  # Debug: List final columns
    
    # Handle binnedInc column: parse only if it contains strings, otherwise keep as is
    if test_df['binnedInc'].dtype == 'object':  # Check if column contains strings
        test_df['binnedInc'] = test_df['binnedInc'].apply(parse_bin)
    else:
        print("binnedInc is already numerical, skipping parse_bin.")
    
    for col in test_df.columns:
        if test_df[col].dtype in ['float64', 'int64'] and test_df[col].isnull().any():
            test_df[col] = test_df[col].fillna(test_df[col].mean())
    
    # Verify the number of features matches the expected input_size
    if test_df.shape[1] != input_size:
        print(f"Error: Expected {input_size} features, but found {test_df.shape[1]}. Please check data preprocessing.")
        return None
    
    # Load the scaler and transform the data
    scaler = joblib.load('scaler.pkl')
    X_test_new = scaler.transform(test_df.values)
    test_tensor = torch.tensor(X_test_new, dtype=torch.float32).to(device)
    
    # Generate predictions
    with torch.no_grad():
        predictions = model(test_tensor).cpu().numpy().flatten()
    print("Predictions:", predictions)
    return predictions

if __name__ == "__main__":
    # Configuration for the best DNN model (DNN-30-8) based on report results
    hidden_layers = [30, 8]  # Architecture of the best model
    model_path = 'best_dnn.pth'  # Path to the saved best model
    test_csv_path = 'test.csv'  # Path to the test dataset
    
    # Run the test function
    test_model(model_path, test_csv_path, hidden_layers)