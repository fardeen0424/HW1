## Submission Overview

This folder contains all files required for the CSE6073 Homework 1 submission. Below is a description of each file and folder included.

### Files and Folders
- **best_dnn.pth**: Saved weights for the best-performing DNN model (DNN-30-8).

- **cancer_reg-1.csv**: Original dataset used for training and evaluation.

- **code.ipynb**: Jupyter notebook containing the complete code for data processing, model training, and evaluation.

- **linear.pth**: Saved weights for the Linear regression model.

- **README.md**: This file, providing instructions and file descriptions.

- **report.pdf**: Detailed report answering all homework questions with tables, plots, and screenshots.

- **requirements.txt**: List of Python dependencies required to run the code.

- **scaler.pkl**: Saved StandardScaler object used for feature preprocessing.

- **test.csv**: Test dataset split (15% of original data).

- **test.py**: Python script (Contains `test_model` function) to load the best DNN model and predict on the test.csv.

- **train.csv**: Training dataset split (70% of original data).

- **val.csv**: Validation dataset split (15% of original data).

- **plots/**: Folder containing 13 loss plot images (e.g., `Linear_loss.png`, `DNN-30-8_loss.png`) with detailed captions showing training and validation loss curves for each model.

- **screenshot_proofs/**: Folder containing 4 screenshot images (e.g., `1.png`, `2.png`, `3.png`, `4.png`) of training and testing outputs with timestamps.

### Testing Process
To evaluate the model predictions on the test set, follow these comprehensive steps:

1. **Unzip the Submission**: Extract the `HW1.zip` file into a folder 

2. **Navigate to the Folder**: Open your preferred code editor (e.g., VS Code, PyCharm) and navigate to the unzipped folder using the terminal or file explorer.

3. **Install Requirements**: Ensure all required Python libraries are installed by running the following commands in the terminal: 
 - `python -m venv venv` → Creates a virtual environment
 - `venv\Scripts\activate` → Activates VENV
 - `pip install -r requirements.txt` → Installs all the dependencies

This will install dependencies such as PyTorch, scikit-learn, matplotlib, numpy, and pandas.

4. **Verify File Presence**: Confirm that all files (`best_dnn.pth`, `scaler.pkl`, `test.csv`, `test.py`) are in the same directory. Do not move or rename these files.

5. **Run the Test Script**: Execute the test script by running the following command in the terminal: `python test.py` This will load the best DNN model (DNN-30-8) and print the predictions for the test set. The output will display an array of predicted TARGET_deathRate values.


