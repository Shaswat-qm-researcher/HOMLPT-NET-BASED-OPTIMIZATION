################################################################
# FUNCTIONS FOR MODEL BUILDING, TRAINING, AND EVALUATION
################################################################

################################################################
# Loading necessary libraries
################################################################

import time
import tensorflow as tf
import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from sklearn.metrics import make_scorer, r2_score, root_mean_squared_error, mean_absolute_error, mean_squared_error, explained_variance_score
from sklearn.base import BaseEstimator, RegressorMixin, clone
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.neural_network import MLPRegressor
import os
from tensorflow.keras.optimizers import AdamW
import warnings
from sklearn.exceptions import DataConversionWarning
import sys
from tensorflow.keras import layers, Model, Input
from tensorflow.keras import layers, models, regularizers, callbacks, optimizers
from tensorflow.keras.callbacks import EarlyStopping
import json

max_processors = os.cpu_count()

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DataConversionWarning)

GREEN_TEXT = "\033[92m"
BLUE_TEXT = "\033[94m" 
RED_TEXT = "\033[91m"
RESET_TEXT = "\033[0m"

################################################################
# Loading Input Data
################################################################


def get_file_from_user():
    supported_extensions = ['.csv', '.xlsx', '.xls', '.txt']

    choice = input("Do you want to load the file from your home directory? (y/n): ").strip().lower()

    if choice == 'y' or choice == 'yes' or choice == 'Y' or choice == 'YES' or choice == 'Yes':
        directory = os.path.expanduser("~") + "/"
    elif choice == 'n' or choice == 'no' or choice == 'N' or choice == 'NO' or choice == 'No':
        directory = input("Enter the full path to the file (use '/' or '\\\\' as needed): ").strip()
        if not directory.endswith(('/', '\\')):
            directory += '/'

    while True:
        file_name = input("Enter the file name *including extension* (e.g., data.xls, data.xlsx, input.csv, test.txt): ").strip()

        if not any(file_name.lower().endswith(ext) for ext in supported_extensions):
            print(f"{RED_TEXT} Invalid file extension. Supported formats: {', '.join(supported_extensions)}{RESET_TEXT}")
            continue

        full_path = os.path.join(directory, file_name)

        if not os.path.exists(full_path):
            print(f"{RED_TEXT} Error: File '{file_name}' does not exist at '{directory}'{RESET_TEXT}")
        else:
            print(f"Loading file from: {full_path}")
            return full_path

def load_data(file_path):
    ext = file_path.lower().split('.')[-1]

    try:
        if ext == 'csv':
            df = pd.read_csv(file_path)
        elif ext in ['xls', 'xlsx']:
            df = pd.read_excel(file_path)
        elif ext == 'txt':
            df = pd.read_csv(file_path, delimiter='\t', engine='python')
        else:
            print(f"{RED_TEXT}Unsupported file format: .{ext}{RESET_TEXT}")
            return None, None, None
    except Exception as e:
        print(f"{RED_TEXT} Failed to load file: {e}{RESET_TEXT}")
        return None, None, None

    # Check for missing values
    if df.isnull().values.any():
        print(f"{RED_TEXT}Error: The file '{file_path}' contains missing values. Exiting the program.{RESET_TEXT}")
        sys.exit(1)

    print(f"{GREEN_TEXT} File loaded successfully. No missing values found.{RESET_TEXT}")

    # Show columns
    print("\n Current columns in the dataset:")
    print(*df.columns.tolist(), sep=", ")

    # Ask if user wants to drop any columns
    drop_choice = input("\nDo you want to drop any columns before selecting inputs/outputs? (y/n): ").strip().lower()
    while drop_choice not in ['y','yes','Y','YES','Yes', 'n', 'No','NO', 'no']:
        print(f"{RED_TEXT}Invalid input. Please enter 'y' or 'n'.{RESET_TEXT}")
        drop_choice = input("Do you want to drop any columns? (y/n): ").strip().lower()

    if drop_choice == 'y' or drop_choice == 'yes' or drop_choice == 'Y' or drop_choice == 'YES' or drop_choice == 'Yes':
        while True:
            try:
                num_cols_to_drop = int(input("How many columns do you want to drop (e.g., 1, 2, 3 .... more): ").strip())
                if num_cols_to_drop <= 0:
                    raise ValueError
                break
            except ValueError:
                print(f"{RED_TEXT}Invalid input. Please enter a positive integer.{RESET_TEXT}")

        cols_to_drop = []
        while True:
            cols_to_drop = input("\nEnter the column names to drop (separated by commas): ").strip().split(",")
            # Strip spaces around each column name
            cols_to_drop = [col.strip() for col in cols_to_drop]

            # Check for invalid columns
            invalid_cols = [col for col in cols_to_drop if col not in df.columns]
            if invalid_cols:
                print(f"{RED_TEXT}Error: The following columns were not found: {', '.join(invalid_cols)}{RESET_TEXT}")
                print("Please enter valid column names from the list above.")
            else:
                break
                
        df.drop(columns=cols_to_drop, inplace=True)
        print(f"{GREEN_TEXT}Successfully dropped columns: {cols_to_drop}{RESET_TEXT}")

    # After dropping, ask user for input and output column names
    print(f"{GREEN_TEXT}\n Updated columns:{RESET_TEXT}")
    print(*df.columns.tolist(), sep=", ")
    
    # Input columns
    while True:
        input_cols = input("Enter input (X) column names separated by commas: ").strip().split(',')
        input_cols = [col.strip() for col in input_cols]
        if all(col in df.columns for col in input_cols):
            break
        else:
            print(f"{RED_TEXT}One or more input columns are invalid. Try again.{RESET_TEXT}")

    # Output column
    while True:
        output_col = input("Enter the output (y) column name: ").strip()
        if output_col in df.columns and output_col not in input_cols:
            break
        else:
            print(f"{RED_TEXT}Invalid or duplicate output column. Try again.{RESET_TEXT}")

    X_data = df[input_cols]
    y_data = df[output_col]
    
    return X_data, y_data, df
    
################################################################
# Creating arrays form the datafile for data preprocessing.
################################################################
data_file = get_file_from_user()

if data_file:
    X, y, NN_df = load_data(data_file)
    if NN_df is not None:
        print(X.head())
Unit = input("Enter the unit for the (Y) column (for no unit leave blank)")
##########################################################################
# Checking if number of parameters are more than datapoint (underfit)
##########################################################################

while True:
    try:
        data_points = X.shape[0]  # Number of data points.
        num_params = X.shape[1]  # Number of parameters.
        
        # Check condition if true, the code will exit with an error.
        if num_params > data_points:
            print(f"{RED_TEXT}Error: Number of parameters ({num_params}) exceeds the number of data points ({data_points}). Please adjust your input datafile. {RESET_TEXT}")
            sys.exit(1)
        
        # Proceed with further processing if datapoints are more than parameters.
        else:
            print(f"{GREEN_TEXT}Data is valid. Proceeding with execution.{RESET_TEXT}")
            break
    
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)
        
################################################################
# Determine the test size based on data points
################################################################

if data_points < 20:
    TEST_SIZE = 0.15
elif 20 <= data_points < 30:
    TEST_SIZE = 0.2
elif 30 <= data_points < 40:
    TEST_SIZE = 0.25
else:
    TEST_SIZE = 0.3

print(f"{BLUE_TEXT}Number of data points: {data_points}{RESET_TEXT}")
print(f"{BLUE_TEXT}Test size selected: {TEST_SIZE}{RESET_TEXT}")

################################################################
# Random state for data processing
################################################################

RANDOM_STATE=42

################################################################
# File path for saving the models and figures
################################################################
print("\n")
save_location_choice = input("Do you want to save the file in the home directory? (y/n): ").strip().lower()
if save_location_choice == 'y' or save_location_choice == 'yes' or save_location_choice == 'Yes' or save_location_choice == 'YES' or save_location_choice == 'Y':
    base_dir = os.path.join(os.path.expanduser('~'), 'DNN')
if save_location_choice == 'n' or save_location_choice == 'no' or save_location_choice == 'No' or save_location_choice == 'NO' or save_location_choice == 'N':
    while True:
        base_dir = input("Enter full directory path where you want to save the file: ").strip()
        if not os.path.exists(base_dir):
            print(f"{RED_TEXT}Directory does not exist! Please provide a valid path.{RESET_TEXT}")
        elif not os.path.isdir(base_dir):
            print(f"{RED_TEXT}Path exists but is not a directory. Try again.{RESET_TEXT}")
        else:
            break

# Step 2: Create 'DNN' subdirectory if it doesn't exist
save_dir = os.path.join(base_dir, "DNN")
try:
    os.makedirs(save_dir, exist_ok=True)
    print(f"{GREEN_TEXT}Directory created or already exists at: {save_dir}{RESET_TEXT}")
except Exception as e:
    print(f"{RED_TEXT}Failed to create 'DNN' folder. Error: {e}{RESET_TEXT}")
    exit(1)

# Step 3: Prompt for valid filename
while True:
    savename = input("Enter the name to save the results in Excel file (without .xlsx extension): ").strip()

    if not savename:
        print(f"{RED_TEXT}Filename cannot be empty. Please enter a valid name.{RESET_TEXT}")
    elif any(char in savename for char in r'\/:*?"<>|'):
        print(f"{RED_TEXT}Invalid characters in filename! Avoid \\ / : * ? \" < > |{RESET_TEXT}")
    else:
        full_path = os.path.join(save_dir, savename + ".xlsx")
        if os.path.exists(full_path):
            overwrite = input(f"{YELLOW_TEXT}File '{savename}.xlsx' already exists in 'DNN'. Overwrite? (y/n): {RESET_TEXT}").strip().lower()
            if overwrite == 'y':
                break
            else:
                print("Please choose a different filename.")
        else:
            break

print(f"{GREEN_TEXT}File will be saved at: {full_path}{RESET_TEXT}")

save_file_path = os.path.join(base_dir, "DNN")
os.makedirs(save_dir, exist_ok=True)


# Full path to save file
save_dir = os.path.join(save_file_path, savename + ".xlsx")

print(f"{GREEN_TEXT}Results will be saved to: {save_dir}{RESET_TEXT}")
start_time_total = time.time()
print("\n")
################################################################
# Function to normalize the Input data using MinMaxScaler
# Set range to (0, 1) for easy convergence
################################################################

def scale_data_x(X):
    scaler_X_NN = StandardScaler()
    X_scaled = scaler_X_NN.fit_transform(X)
    return X_scaled, scaler_X_NN

################################################################
# Function to normalize the Output data using MinMaxScaler
# Set range to (0, 1) for easy convergence
################################################################

def scale_data_y(y):
    scaler_y_NN = StandardScaler()
    y_scaled = scaler_y_NN.fit_transform(y.values.reshape(-1, 1))
    return y_scaled, scaler_y_NN


################################################################
# Scaling data
################################################################

X_scaled, scaler_X_NN = scale_data_x(X)
y_scaled, scaler_y_NN = scale_data_y(y)

######################################################################################
# Function for splitting training and testing data based on predetermined test size
######################################################################################

def split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

################################################################
# Spliting data for training and testing
################################################################

X_train, X_test, y_train, y_test = split_data(X_scaled, y_scaled)

################################################################
# Customized loss function for weighted loss
################################################################

# Custom log-cosh Loss function.
def huber_loss(y_true, y_pred):
    delta=1.0
    error = y_pred - y_true
    abs_error = np.abs(error)
    
    # Quadratic part for small errors (|y - y_pred| <= delta)
    quadratic_part = np.minimum(abs_error, delta)
    
    # Linear part for large errors (|y - y_pred| > delta)
    linear_part = abs_error - quadratic_part
    
    # Calculate final Huber loss
    loss = 0.5 * quadratic_part ** 2 + delta * linear_part
    return np.mean(loss)

def quantile_loss(y_true, y_pred):
    tau=0.7
    # Calculate the residuals (differences between true and predicted values)
    residuals = y_true - y_pred
    
    # Apply quantile loss with emphasis on under-predictions
    loss = np.where(residuals > 0, (1 - tau) * residuals, tau * (-residuals))
    
    # Return the mean loss
    return np.mean(loss)

# R2 score function.
def R2(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    return r2

# Custom weighted loss metric.
def custom_loss(y_true, y_pred):
    
    # Compute individual metrics.
    huber_loss_cl = huber_loss(y_true, y_pred)
    rmse_cl = root_mean_squared_error(y_true, y_pred)
    quantile_loss_cl =  quantile_loss(y_true, y_pred)
    r2_cl = r2_score(y_true, y_pred)

    # Bounded metrics.
    huber_loss_bounded = np.abs(huber_loss_cl) / (np.abs(huber_loss_cl) + 1)  # Scale to [0, 1).
    rmse_bounded = np.abs(rmse_cl) / (np.abs(rmse_cl) + 1)              # Scale to [0, 1).
    quantile_loss_bounded = np.abs(quantile_loss_cl) / (np.abs(quantile_loss_cl) + 1)                 # Scale to [0, 1).
    
    huber_loss_reversed = 1 - huber_loss_bounded
    rmse_reversed = 1 - rmse_bounded
    quantile_loss_reversed = 1 - quantile_loss_bounded
    
    # Weighted sum of bounded metrics.
    final_score =  0.2 * quantile_loss_reversed + 0.28 * rmse_reversed + 0.2 * huber_loss_reversed + 0.28 * r2_cl 
    
    return final_score

################################################################
# Funtion to calculate total number of hidden layers
################################################################

def calculate_total_neurons(layers):
    return sum(layers)

################################################################
# User Input for Hyperparameters
################################################################

while True:
    user_choice = input("Would you like to manually enter hyperparameters or use automatic tuning with GridSearchCV? (enter 'manual' or 'auto'): ").strip().lower()
    
    if user_choice == 'manual':
        
        # Displaying important note before collecting inputs.
        print("\033[1mNote: These hyperparameters are defined as per the problem and data complexity.\033[0m\n")
        print("\033[1mChanging these may cause variations in results.\033[0m\n")
        print("\033[1mALL VALUES ARE CASE SENSITIVE.\033[0m\n")

        # Function to validate input as an integer within a specified range.
        def get_integer_input(prompt, min_val, max_val, preferred_val=None):
            while True:
                try:
                    value = input(f"{prompt} (Range: {max_val}-{min_val}, Can try: {preferred_val}): ")
                    value = int(value)
                    if min_val <= value <= max_val:
                        return value
                    else:
                        print(f"{RED_TEXT}Please enter a value between {min_val} and {max_val}.{RESET_TEXT}")
                except ValueError:
                    print(f"{RED_TEXT}Invalid input. Please enter an integer.{RESET_TEXT}")

        # Function to validate input as a float within a specified range.
        def get_float_input(prompt, min_val, max_val, preferred_val=None):
            while True:
                try:
                    value = input(f"{prompt} (Range: {max_val}-{min_val}, Can try: {preferred_val}): ")
                    value = float(value)
                    if min_val <= value <= max_val:
                        return value
                    else:
                        print(f"{RED_TEXT}Please enter a value between {min_val} and {max_val}.{RESET_TEXT}")
                except ValueError:
                    print(f"{RED_TEXT}Invalid input. Please enter a float.{RESET_TEXT}")

        # Function for hidden layers.
        def get_hidden_layers():
            hidden_layers = []
            
            # First mandatory hidden layer.
            first_layer = get_integer_input(
                "Enter number of neurons for the first hidden layer (Can try: 128)", 
                4, 4096, preferred_val=128
            )
            hidden_layers.append(first_layer)
            
            # Optional second and third layers..
            print("Enter number of neurons for the second and third hidden layers in descending order (multiples of 2, max 4096, ending at any multiple of 2 except 2). Each layer is optional.")
            
            preferred_layers = [64, 32]  # Preferred values for the second and third layers.
            for i in range(2):  # Loop for the next two layers.
                while True:
                    cont = input(f"Do you want to add layer {i+2} (Can try: {preferred_layers[i]} if available)? (y/n): ").strip().lower()
                    if cont not in ['y', 'n']:
                        print(f"{RED_TEXT}Invalid input. Please enter 'y' or 'n'.{RESET_TEXT}")
                    else:
                        break
                if cont == 'n':
                    break
                while True:
                    layer = get_integer_input(
                        f"Enter number of neurons for layer {i+2}",
                        4, 4096, preferred_val=preferred_layers[i]
                    )
                    if layer == 2:
                        print(f"{RED_TEXT}Number of neurons in hidden-layer cannot be 2. Please enter a different value.{RESET_TEXT}")
                    elif layer >= hidden_layers[-1]:
                        print(f"{RED_TEXT}Please enter a smaller value than the previous layer ({hidden_layers[-1]}).{RESET_TEXT}")
                    else:
                        hidden_layers.append(layer)
                        break

            # Additional layers beyond the third layer.
            if len(hidden_layers) >= 3:
                while True:
                    while True:
                        cont = input("Do you want to add another layer? (y/n): ").strip().lower()
                        if cont not in ['y','yes','Y','YES','Yes', 'n', 'No','NO', 'no']:
                            print(f"{RED_TEXT}Invalid input. Please enter 'y' or 'n'.{RESET_TEXT}")
                        else:
                            break
                    if cont == 'n'or cont == 'No' or cont == 'NO' or cont == 'no':
                        break
                    while True:
                        layer = get_integer_input("Enter number of neurons for the layer: ", 4, 4096)
                        if layer == 2:
                            print(f"{RED_TEXT}Number of neurons in hidden-layer cannot be 2. Please enter a different value.{RESET_TEXT}")
                        elif layer >= hidden_layers[-1]:
                            print(f"{RED_TEXT}Please enter a smaller value than the previous layer ({hidden_layers[-1]}).{RESET_TEXT}")
                        else:
                            hidden_layers.append(layer)
                            break
            return hidden_layers

        # Collecting other parameters from user.
        
        LEARNING_RATE = get_float_input("Enter the learning rate", 0.00001, 0.5, preferred_val=0.01)
        NUM_EPOCHS = get_integer_input("Enter the number of epochs (iterations)", 50, 1000, preferred_val=100)
        BATCH_SIZE = get_integer_input("Enter the batch size", 2, 512, preferred_val=2)
        RANDOM_STATE = get_integer_input("Enter the random state",2,314, preferred_val=42)
        
        # Finalize hidden layers based on user input.
        HIDDEN_LAYERS = get_hidden_layers()

        # Display the chosen hyperparameters.
        print(f"{BLUE_TEXT} \n Chosen Hyperparameters: {RESET_TEXT}")
        print(f"{BLUE_TEXT} Hidden Layers With Neurons: {HIDDEN_LAYERS} {RESET_TEXT}")
        print(f"{BLUE_TEXT} Learning Rate: {LEARNING_RATE} {RESET_TEXT}")
        print(f"{BLUE_TEXT} Number of Epochs (iterations): {NUM_EPOCHS} {RESET_TEXT}")
        print(f"{BLUE_TEXT} Batch Size: {BATCH_SIZE} {RESET_TEXT}")
        print(f"{BLUE_TEXT} Test Size: {TEST_SIZE} {RESET_TEXT}")
        print("\n")
        break

    elif user_choice == 'auto':
        
        # Determining the number of processor for grid search.
        while True:
            try:
                num_processors = int(input("Input the number of CPU processors you want to use for parallel processing (For all use -1): "))
                if num_processors == -1 or (num_processors > 0 and num_processors <= max_processors):
                    break
                else:
                    if num_processors > max_processors:
                        print(f"{RED_TEXT}Error: The number exceeds the available processors ({max_processors}).{RESET_TEXT}")
                    else:
                        print(f"{RED_TEXT}Error: Please enter -1 or an integer greater than 0.{RESET_TEXT}")
            except ValueError:
                print(f"{RED_TEXT}Error: Invalid input. Please enter an integer.{RESET_TEXT}")
            
        print(f"{GREEN_TEXT}\nAutomatically selecting NN hyperparameters with GridSearchCV...{RESET_TEXT}")

        #######################
        # GridSearchCV Setup.
        #######################
        class TrackedMLP(BaseEstimator, RegressorMixin):
            def __init__(self, **kwargs):
                self.model = MLPRegressor(**kwargs)
                self.n_iter_ = None
                self.loss_curve_ = None
    
            def fit(self, X, y):
                self.model.fit(X, y)
                self.n_iter_ = self.model.n_iter_
                self.loss_curve_ = self.model.loss_curve_
                return self
    
            def predict(self, X):
                return self.model.predict(X)
    
            def get_params(self, deep=True):
                return self.model.get_params(deep)
    
            def set_params(self, **params):
                self.model.set_params(**params)
                return self
        
        scorings = {
            'Huber loss': make_scorer(huber_loss, greater_is_better=False),
            'Quantile loss': make_scorer(quantile_loss, greater_is_better=False),
            'RMSE': make_scorer(root_mean_squared_error, greater_is_better=False),
            'R2': make_scorer(R2, greater_is_better=True),
            'Custom_Loss': make_scorer(custom_loss, greater_is_better=True),
        }
        
        #######################
        # Parameter Grid Blocks
        #######################
        
        param_grid = [
        # Block 1: Small to Medium architectures + small batch + small LR
        {
        'hidden_layer_sizes': [
            (8,), (16,), (64,), (128,), 
            (64, 32), (128, 64)
        ],
        'learning_rate_init': [0.001, 0.0001],
        'batch_size': [8, 16, 32],
        'max_iter': [150, 250, 350]
        },

        # Block 2: Small to Medium architectures + large batch + larger LR
        {
        'hidden_layer_sizes': [
            (16,), (64,), (128,), 
            (64, 32), (128, 64)
        ],
        'learning_rate_init': [0.01, 0.1],
        'batch_size': [64, 128, 256],
        'max_iter': [150, 250, 350]
        },

        # Block 3: Medium-Deep architectures + small batch + small LR
        {
        'hidden_layer_sizes': [
            (256,), (512,), (512, 256),
            (512, 256, 128)
        ],
        'learning_rate_init': [0.0001, 0.001],
        'batch_size': [8, 16, 32],
        'max_iter': [300, 400, 500]
        },

        # Block 4: Medium-Deep architectures + large batch + larger LR
        {
        'hidden_layer_sizes': [
            (256,), (512,), (512, 256),
            (512, 256, 128)
        ],
        'learning_rate_init': [0.01, 0.1],
        'batch_size': [64, 128, 256],
        'max_iter': [300, 400, 500]
        },

        # Block 5: Deep architectures + small batch + small LR
        {
        'hidden_layer_sizes': [
            (1024,), (2048,),
            (1024, 512, 256), (2048, 1024, 512),
            (512, 256, 128, 64)
        ],
        'learning_rate_init': [0.0001, 0.001],
        'batch_size': [8, 16, 32],
        'max_iter': [400, 500, 700, 1000]
        },

        # Block 6: Deep architectures + large batch + large LR
        {
        'hidden_layer_sizes': [
            (1024,), (2048,),
            (1024, 512, 256), (2048, 1024, 512),
            (512, 256, 128, 64)
        ],
        'learning_rate_init': [0.01, 0.1],
        'batch_size': [64, 128, 256],
        'max_iter': [400, 500, 700, 1000]
        }
         ]

        ##############################################
        # Model Setup for MLP regressor.
        ##############################################
        
        model_1 = TrackedMLP(random_state = RANDOM_STATE, early_stopping=True)
        
        ##############################################
        # Performing Grid Search CV for each block.
        ##############################################
        
        grid_search = GridSearchCV(
            estimator=model_1, 
            param_grid=param_grid, 
            scoring=scorings,  # Use custom scoring function
            refit="Custom_Loss",
            cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
            n_jobs=num_processors, 
            verbose=3,
            return_train_score=True
            )

        # Progress Tracking for GridSearchCV.
        print("Starting Grid Search...")
        
        grid_search.fit(X_scaled, y_scaled.ravel())

        # Extract best model (if early-stopped)
        best_model = grid_search.best_estimator_
        
        # Actual early-stopped iteration count and loss curve from wrapped model
        if hasattr(best_model, "n_iter_"):
            actual_iter = best_model.n_iter_
            print(f"\n Best model stopped after {actual_iter} iterations.")

        if hasattr(best_model, "loss_curve_"):
            final_loss = best_model.loss_curve_[-1] if best_model.loss_curve_ else None
            print(f" Final training loss after early stopping: {final_loss:.6f}")
        
        
        joblib.dump(grid_search, os.path.join(save_file_path, 'grid_search_checkpoint.pkl'))
        
        cv_results = grid_search.cv_results_
        
        ###############################################################
        # Saving results to dataframe and filterring overfitting
        ###############################################################
        # Access the complete results in DataFrame format and Saved results for further analysis.
        # List of metrics to correct in Excel file, as grid search include -ve sign infront of minimization metrices.
        results_df = pd.DataFrame(grid_search.cv_results_)
        results_df['best_n_iter'] = actual_iter
        results_df['best_final_loss'] = final_loss
        
        # Overfitting threshold 
        overfit_threshold = 0.07

        results_df['train_test_gap'] = (results_df['mean_train_R2'] - results_df['mean_test_R2']).abs()

        # Filter non-overfitted results
        filtered_results_df = results_df[results_df['train_test_gap'] < overfit_threshold]

        if filtered_results_df.empty:
            print("All models appear overfitted. Using original best model from GridSearchCV.")
            best_model = grid_search.best_estimator_
            best_params_1 = grid_search.best_params_
            best_loss = grid_search.best_score_
        
        else:
            # Pick best from filtered (non-overfit) results
            best_row = filtered_results_df.sort_values(by='mean_test_Custom_Loss', ascending=False).iloc[0]
            best_params_1 = best_row['params']
            best_loss = best_row['mean_test_Custom_Loss']

        
        # Save the best parameters to specific variables.
        LEARNING_RATE = best_params_1['learning_rate_init']
        NUM_EPOCHS = best_params_1['max_iter']
        BATCH_SIZE = best_params_1['batch_size']
        HIDDEN_LAYERS = best_params_1['hidden_layer_sizes']
        
        print(f"{BLUE_TEXT}\n Best hyperparameters test: {best_params_1}{RESET_TEXT}")
        print(f"{BLUE_TEXT}\n Best custom loss score test: {np.abs(best_loss)}{RESET_TEXT}")
        
                        
        
        # For Examples (columns containing "Huber loss", "Quantile loss", "RMSE").
        error_metrics = [col for col in results_df.columns if any(err in col for err in ["Huber loss", "Quantile loss", "RMSE"])]

        # Convert negative error values to positive (absolute values).
        results_df[error_metrics] = results_df[error_metrics].abs()
        results_df.to_excel(os.path.join(save_file_path, f'{savename}_GridSearch_results.xlsx'),index=True)
        filtered_results_df.to_excel(os.path.join(save_file_path, f'{savename}_GridSearch_Filtered_NoOverfit.xlsx'), index=False)
        pd.DataFrame({'loss': best_model.loss_curve_}).to_excel(os.path.join(save_file_path, f'{savename}_BestModel_LossCurve.xlsx'), index_label='Epoch')
        ##################################################################
        # Training MLP with best hyperparameter found from Grid Search.
        ##################################################################
        
        start_time_total_MLP = time.time()        
        optimized_model = MLPRegressor(
            hidden_layer_sizes=HIDDEN_LAYERS,
            learning_rate_init=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            max_iter=NUM_EPOCHS,
            random_state=RANDOM_STATE
        )
        start_time_1_MLP=time.time()
        optimized_model.fit(X_train, y_train)
        end_time_1_MLP=time.time()
        elapsed_time_1_MLP= end_time_1_MLP - start_time_1_MLP
        
        

        start_time_2_MLP=time.time()
        y_train_pred_MLP = optimized_model.predict(X_train)
        end_time_2_MLP=time.time()
        elapsed_time_2_MLP= end_time_2_MLP - start_time_2_MLP
        
        start_time_3_MLP =time.time()
        y_test_pred_MLP = optimized_model.predict(X_test)
        end_time_3_MLP = time.time()
        elapsed_time_3_MLP = end_time_3_MLP - start_time_3_MLP
        
        y_actual_train_orig_MLP = scaler_y_NN.inverse_transform(y_train.reshape(1, -1))
        y_predicted_train_orig_MLP = scaler_y_NN.inverse_transform(y_train_pred_MLP.reshape(1, -1))
        y_actual_test_orig_MLP = scaler_y_NN.inverse_transform(y_test.reshape(1, -1))
        y_predicted_test_orig_MLP = scaler_y_NN.inverse_transform(y_test_pred_MLP.reshape(1, -1))

        n_train_MLP = y_actual_train_orig_MLP.size
        n_test_MLP = y_actual_test_orig_MLP.size
        
        total_layers_MLP = 1 + len(HIDDEN_LAYERS) + 1  # 1 input layer + hidden layers + 1 output layer.
        total_neurons_MLP = calculate_total_neurons(HIDDEN_LAYERS)
    
        num_iterations_MLP = (n_train_MLP) * NUM_EPOCHS
    
        # Calculating errors for the model.
         
        huber_loss_train_MLP = huber_loss(y_actual_train_orig_MLP, y_predicted_train_orig_MLP)
        huber_loss_test_MLP = huber_loss(y_actual_test_orig_MLP, y_predicted_test_orig_MLP)
        rmse_train_MLP = root_mean_squared_error(y_actual_train_orig_MLP, y_predicted_train_orig_MLP)
        rmse_test_MLP = root_mean_squared_error(y_actual_test_orig_MLP, y_predicted_test_orig_MLP)
        quantile_loss_train_MLP = quantile_loss(y_actual_train_orig_MLP, y_predicted_train_orig_MLP)
        quantile_loss_test_MLP = quantile_loss(y_actual_test_orig_MLP, y_predicted_test_orig_MLP)
        R2_score_train_MLP = R2(y_train, y_train_pred_MLP)
        R2_score_test_MLP = R2(y_test, y_test_pred_MLP)
        custom_loss_train_MLP = custom_loss(y_train, y_train_pred_MLP)
        custom_loss_test_MLP = custom_loss(y_test, y_test_pred_MLP)
        
        end_time_total_MLP = time.time()
        elapsed_time_total_MLP = end_time_total_MLP - start_time_total_MLP
        
        # Combine training and test results
        train_df = pd.DataFrame({
            'Actual_Train': y_actual_train_orig_MLP.flatten(),
            'Predicted_Train': y_predicted_train_orig_MLP.flatten()
        })

        test_df = pd.DataFrame({
            'Actual_Test': y_actual_test_orig_MLP.flatten(),
            'Predicted_Test': y_predicted_test_orig_MLP.flatten()
        })

        # Save to Excel with two sheets
        excel_path = os.path.join(save_file_path, savename + '_MLP_Actual_vs_Predicted_Data.xlsx')
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            train_df.to_excel(writer, sheet_name='Train', index=False)
            test_df.to_excel(writer, sheet_name='Test', index=False)

        print(f"Saved train/test Actual vs Predicted values to {excel_path}")

        # Ploting actual vs train graphs for MLP and dynamically compute figure size.
        plt.figure(figsize=(8.5, 5))

        # Adjust sizes dynamically
        total_points = n_train_MLP + n_test_MLP
        min_scale = 0.5 
        max_scale = 2.5 
        base_size = 80
        scale_factor_MLP = np.clip(1000 / (total_points + 1), min_scale, max_scale)
        
        s_size = 80 * scale_factor_MLP
        
        plt.scatter(y_actual_train_orig_MLP.flatten(), y_predicted_train_orig_MLP.flatten(), color='red', edgecolor='black', s = s_size,
                marker ='o', label=fr'Training Data')
        plt.scatter(y_actual_test_orig_MLP.flatten(), y_predicted_test_orig_MLP.flatten(), marker ='^', s = s_size,
                    label=fr'Test Data', color='none', edgecolor='black')
        plt.plot([min(y), max(y)], [min(y), max(y)], color='black', alpha=0.8, linestyle='--', linewidth='1.3', label=fr'Actual Values')    
   
        
        plt.annotate(fr'R$^2$ Score (Train): {R2_score_train_MLP:.5f}', (0.03, 0.73), 
                     fontsize=11.5, xycoords='axes fraction')
        plt.annotate(fr'R$^2$ Score (Test): {R2_score_test_MLP:.5f}', (0.03, 0.68), 
                     fontsize=11.5, xycoords='axes fraction')
        plt.annotate(fr'Quantile loss (Train): {quantile_loss_train_MLP:.2e}  (Test): {quantile_loss_test_MLP:.2e}', 
             (0.03, 0.93), fontsize=11.5,  xycoords='axes fraction')
        plt.annotate(fr'Huber loss (Train): {huber_loss_train_MLP:.2e}  (Test): {huber_loss_test_MLP:.2e}', 
             (0.03, 0.88), fontsize=11.5,  xycoords='axes fraction')
        plt.annotate(fr'RMSE (Train): {rmse_train_MLP:.2e}  (Test): {rmse_test_MLP:.2e}', 
             (0.03, 0.83), fontsize=11.5,  xycoords='axes fraction')
        plt.annotate(fr'$L_c$ (Train): {custom_loss_train_MLP:.2e}  (Test): {custom_loss_test_MLP:.2e}', 
             (0.03, 0.78), fontsize=11.5,  xycoords='axes fraction')
        
        box_props = dict(boxstyle='round,pad=0.4', edgecolor='black', facecolor='white', alpha=1)
        textstr = '\n'.join((
            fr'MLP Hyperparameters',
            fr'',
            fr'Input Dimension: {num_params}',
            fr'Total Samples: {n_test_MLP + n_train_MLP}',
            fr'Train Samples: {n_train_MLP}',
            fr'Test Samples: {n_test_MLP}',
            fr'Total NN Layers: {total_layers_MLP}',
            fr'Total Neurons: {total_neurons_MLP}',
            fr'Learning Rate: {LEARNING_RATE}',
            fr'No. of Iterations: {NUM_EPOCHS}',
            fr'Random State: {RANDOM_STATE}',
            fr'Training time: {elapsed_time_1_MLP:.3f} s',
            fr'Inference time (Train): {elapsed_time_2_MLP:.3f} s',
            fr'Inference time (Test): {elapsed_time_3_MLP:.3f} s',
            fr'Total time taken: {elapsed_time_total_MLP:.3f} s'
            ))
        plt.annotate(textstr, (1.03, 0.20), fontsize=10, xycoords='axes fraction', bbox=box_props, color='black')
        
        
        plt.xlabel(fr'Actual Values ({y.name} [{Unit}])', fontsize=16)
        plt.ylabel(f'Predicted Values ({y.name} [{Unit}])', fontsize=16)
        #plt.title(fr'{y.name} predictions from MLP', fontsize=14, fontweight='bold')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.tick_params(axis='both', which='both', direction='in', labelsize=17, length=8,
               top=True, bottom=True, left=True, right=True, width=1.5)
        plt.tick_params(axis='both', which='minor', direction='in',
               length=4, width=1.5, top=True, bottom=True, left=True, right=True)
        ax = plt.gca()  # get current axes
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(AutoMinorLocator(10))
        ax.yaxis.set_minor_locator(AutoMinorLocator(7))
    
        plt.legend(loc='lower right', fontsize=12, facecolor='white', edgecolor='black', frameon=True, bbox_to_anchor=(0.97, 0.03))
        plt.grid(False)       
        plt.tight_layout()
        
        # Saving the figure 
        plt.savefig(os.path.join(save_file_path, savename + '_MLP_Actual_vs_Prediceted_Data.jpg'), dpi=800)

        
        # Plot each metric over the course of the iterations (changing hyperparameters).
        metrics = ['mean_test_Huber loss','mean_test_Quantile loss', 'mean_test_RMSE', 'mean_test_R2', 'mean_test_Custom_Loss']
        metrics_0 = ['mean_train_Huber loss','mean_train_Quantile loss', 'mean_train_RMSE', 'mean_train_R2', 'mean_train_Custom_Loss']
        metrics_1 = ['Huber loss','Quantile loss', 'RMSE', 'R2', 'Custom_Loss']
        metric_labels = ['Huber Loss','Quantile Loss', 'Root Mean Squared Error (RMSE)', '$R^2$ Score', 'Custom Loss ($L_c$)']
        metric_labels_1 = ['Huber Loss', 'Quantile Loss', 'RMSE', '$R^2$ Score', '$L_c$'] 
        metric_labels_2 = ['Huber Loss','Quantile Loss', 'RMS Error', 'R2 Score', 'Custom Loss']
        colors = ['lime', 'blue', 'orange', 'magenta', 'red']
        
        markers = ['o', 'o', 'o', 'o', 'o'] 
        higher_is_better = [False, False, False, True, True]
        
        for metric, metric_0, metric_1, label, label_1, color, marker, is_higher_better, saving_name in zip(metrics, metrics_0, metrics_1, metric_labels, metric_labels_1, colors, markers, higher_is_better, metric_labels_2):
            
            optimum_score_index = results_df[f'rank_test_{metric_1}'].argmin()  # Index of the best score for each metric.
            optimum_params = results_df['params'][optimum_score_index]  # Best params corresponding to that score.
            optimum_score = results_df[f'mean_test_{metric_1}'][optimum_score_index]  # Best score for that metric.
            
            LEARNING_RATE_1 = optimum_params.get('learning_rate_init')
            NUM_EPOCHS_1 = optimum_params.get('max_iter')
            BATCH_SIZE_1 = optimum_params.get('batch_size')
            HIDDEN_LAYERS_1 = optimum_params.get('hidden_layer_sizes')

            # Dynamic plot size calculation
            base_width, base_height = 18, 6
            scale_w = 1 + 0.015 * (total_neurons_MLP // 10)
            scale_h = 1 + 0.015 * total_layers_MLP

            fig_width = min(14, base_width * scale_w)
            fig_height = min(10, base_height * scale_h)
            plt.figure(figsize=(fig_width, fig_height))

            # Calculate a scale factor relative to base size
            scale_factor = (fig_width * fig_height) / (base_width * base_height)

            # Adjust sizes dynamically
            marker_size = 10 * scale_factor
            marker_edge_width = 1.7 * scale_factor
            line_width_test = 1.5 * scale_factor
            line_width_train = 2 * scale_factor
            tick_fontsize = 22 * scale_factor
            label_fontsize = 26 * scale_factor
            legend_fontsize = 20 * scale_factor
            annot_fontsize = 14 * scale_factor
            hline_vline_width = 2.2 * scale_factor

            # Plotting
            plt.plot(results_df.index + 1, results_df[metric],
                     marker=marker, linestyle='-.', linewidth=line_width_test,
                     markersize=marker_size, markeredgecolor='black', markeredgewidth=marker_edge_width,
                     markerfacecolor=color, color='black', label='test', zorder=1)

            plt.plot(results_df.index + 1, results_df[metric_0],
                     marker='v', linestyle='-', linewidth=line_width_train, alpha=0.9,
                     markersize=marker_size * 1.1, markeredgecolor='black', markeredgewidth=marker_edge_width * 0.8,
                     markerfacecolor='white', color='black', label='train', zorder=0)

            # Best value and line indicators
            if is_higher_better:
                best_index = results_df[metric].idxmax()
                best_value = results_df[metric].max()
            else:
                best_index = results_df[metric].idxmin()
                best_value = results_df[metric].min()

            plt.axhline(best_value, color='red', linestyle='--', linewidth=hline_vline_width,
                        label=f'Best {label}: {best_value:.4f}', zorder=2)
            plt.axvline(best_index + 1, color='red', linestyle='--', linewidth=hline_vline_width, zorder=2)


            plt.xlabel("Iteration from Grid Search CV (Hyperparameter Combination)", fontsize=label_fontsize)
            plt.ylabel(label, fontsize=label_fontsize)
            plt.xticks(fontsize=tick_fontsize)
            plt.yticks(fontsize=tick_fontsize)

            # Legend
            if metric_1 in ['Huber loss', 'Quantile loss', 'RMSE']:
                legend_loc = 'lower right'
            elif metric_1 in ['R2', 'Custom_Loss']:
                legend_loc = 'upper right'

            # Plot legend
            plt.legend(
                [f"{'Test'}", f"{'Train'}", f"Best {label_1}: {best_value:.4f}"],
                loc=legend_loc,
                fontsize=legend_fontsize,
                facecolor='white', edgecolor='black'
            )
            
            # Annotation box
            box_props = dict(
                boxstyle=f'round,pad={0.5 * scale_factor:.2f}',
                edgecolor='black',
                facecolor='white',
                alpha=1
            )
            if metric_1 in ['Huber loss', 'Quantile loss', 'RMSE']:
                ann_xy = (0.02, 0.03)#(0.02, 0.97)  
                ann_va = 'bottom'#'top'
            elif metric_1 in ['R2', 'Custom_Loss']:
                ann_xy = (0.02, 0.97)   #(0.02, 0.03)  
                ann_va = 'top'#'bottom'
            
            textstr = '\n'.join((
                fr'Best Hyperparameters ({label_1})',
                fr'',
                fr'Learning Rate: {LEARNING_RATE_1}',
                fr'Number of Iterations: {NUM_EPOCHS_1}',
                fr'Batch Size: {BATCH_SIZE_1}',
                fr'Hidden Layer: {HIDDEN_LAYERS_1}'
            ))

            plt.annotate(
                textstr,
                xy=ann_xy,  
                fontsize=annot_fontsize * 1.25,
                xycoords='axes fraction',
                bbox=box_props,
                verticalalignment=ann_va,
                horizontalalignment='left',
                color='black'
            )

            plt.tick_params(axis='both', which='both', direction='in', labelsize=20, length=8,
                       top=True, bottom=True, left=True, right=True, width=1.5)
            plt.tick_params(axis='both', which='minor', direction='in',
                       length=4, width=1.5, top=True, bottom=True, left=True, right=True)
            ax = plt.gca()  # get current axes
            ax.minorticks_on()
            if metric_1 in ['Huber loss', 'Quantile loss']:
                ax.xaxis.set_minor_locator(AutoMinorLocator(20))
                ax.yaxis.set_minor_locator(AutoMinorLocator(10))
            elif metric_1 in ['RMSE']:
                ax.xaxis.set_minor_locator(AutoMinorLocator(20))
                ax.yaxis.set_minor_locator(AutoMinorLocator(7))
            elif metric_1 in ['R2', 'Custom_Loss']:
                ax.xaxis.set_minor_locator(AutoMinorLocator(20))
                ax.yaxis.set_minor_locator(AutoMinorLocator(7))
                
            if metric_1 in ['Huber loss', 'Quantile loss']:
                plt.ylim(-0.2,0.3)
            elif metric_1 in ['RMSE']:
                plt.ylim(-0.3,0.5)
            elif metric_1 in ['R2', 'Custom_Loss']:
                plt.ylim(0,1.5)

            
            # Final layout and save
            plt.tight_layout()
            plt.savefig(os.path.join(save_file_path, f"{savename}_{saving_name.replace(' ', '_')}.jpg"), dpi=800)
        print("\n")
        
                               
        break

    else:
        print(f"{RED_TEXT}Invalid input. Please enter 'manual' or 'auto' to continue.{RESET_TEXT}")

print(f"{GREEN_TEXT} Grid Search completed.{RESET_TEXT}")
print("\n")
################################################################
# Deep Neural Network model building
################################################################

################################################################
# Function for DNN model building with optimized hyperparameters
################################################################

def build_model(input_shape):
    model = models.Sequential()

    # First layer
    model.add(layers.Dense(
        HIDDEN_LAYERS[0], 
        input_shape=(input_shape,),
        kernel_initializer='glorot_uniform',
        kernel_regularizer=regularizers.l2(0.0001)
    ))
    model.add(layers.PReLU())

    # Hidden layers
    for layer_size in HIDDEN_LAYERS[1:]:
        model.add(layers.Dense(
            layer_size,
            kernel_initializer='glorot_uniform',
            kernel_regularizer=regularizers.l2(0.0001)
        ))
        model.add(layers.PReLU())

    # Output layer for regression (no activation)
    model.add(layers.Dense(1))
    
    return model

################################################################
# Function for Training the DNN model
################################################################
def compile_and_train(model, X_train, y_train, X_test, y_test, 
                      learning_rate=LEARNING_RATE, num_epochs=NUM_EPOCHS, 
                      batch_size=BATCH_SIZE):

    optimizer = optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-5)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # # Early stopping callback
    # early_stop = callbacks.EarlyStopping(
    #     monitor='val_loss',
    #     min_delta=1e-12,       # Minimum change in val_loss to qualify as improvement
    #     patience=40,          # Stop if no improvement for 10 epochs
    #     restore_best_weights=True
    # )
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=num_epochs,
        batch_size=batch_size,
        verbose=1#,
        # callbacks=[early_stop]
    )
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    print(f"Training time: {elapsed_time:.2f} seconds")
    return history, elapsed_time


################################################################
# Function for calculating error in the prediction from DNN
################################################################

def evaluate_model(model, X_test, y_test, X_train, y_train):
    start_time = time.time()
    predicted_train = model.predict(X_train, verbose=0)
    end_time = time.time()
    elapsed_time_2 = end_time - start_time
    print(f"Inference time (Train): {elapsed_time_2:.2f} seconds")
    
    start_time = time.time()
    predicted_test = model.predict(X_test, verbose=0)
    end_time = time.time()
    elapsed_time_3 = end_time - start_time
    print(f"Inference time (Test): {elapsed_time_3:.2f} seconds")
    
    y_actual_train_orig = scaler_y_NN.inverse_transform(y_train.reshape(1, -1))
    y_predicted_train_orig = scaler_y_NN.inverse_transform(predicted_train.reshape(1, -1))
    y_actual_test_orig = scaler_y_NN.inverse_transform(y_test.reshape(1, -1))
    y_predicted_test_orig = scaler_y_NN.inverse_transform(predicted_test.reshape(1, -1))

    
    huber_loss_train_DNN = huber_loss(y_actual_train_orig, y_predicted_train_orig)
    huber_loss_test_DNN = huber_loss(y_actual_test_orig, y_predicted_test_orig)
    rmse_train_DNN = root_mean_squared_error(y_actual_train_orig, y_predicted_train_orig)
    rmse_test_DNN = root_mean_squared_error(y_actual_test_orig, y_predicted_test_orig)
    quantile_loss_train_DNN = quantile_loss(y_actual_train_orig, y_predicted_train_orig)
    quantile_loss_test_DNN = quantile_loss(y_actual_test_orig, y_predicted_test_orig)
    R2_score_train_DNN = R2(y_train, predicted_train)
    R2_score_test_DNN = R2(y_test, predicted_test)
    custom_loss_train_DNN = custom_loss(y_train, predicted_train)
    custom_loss_test_DNN = custom_loss(y_test, predicted_test)
    
    return elapsed_time_2, elapsed_time_3, huber_loss_train_DNN, huber_loss_test_DNN, rmse_train_DNN, rmse_test_DNN, quantile_loss_train_DNN, quantile_loss_test_DNN, R2_score_train_DNN, R2_score_test_DNN, custom_loss_train_DNN, custom_loss_test_DNN, predicted_train, predicted_test

###################################################################################
# Function to Save the DNN model for further use (like during optimization)
###################################################################################

def save_model(model, scaler_X_NN, scaler_y_NN, save_file_path, model_name):
    save_folder = os.path.join(save_file_path, savename)
    os.makedirs(save_folder, exist_ok=True) 
    
    model_path = os.path.join(save_folder, f"{savename}_{model_name}.keras")
    model.save(model_path)

###################################################################################
# Function to obtain prediction from DNN in an excel file with original scale
###################################################################################

def save_results_to_excel(y_actual_train, predicted_train, y_actual_test, predicted_test, huber_loss_train_DNN, huber_loss_test_DNN, rmse_train_DNN, rmse_test_DNN, quantile_loss_train_DNN, quantile_loss_test_DNN, R2_score_train_DNN, R2_score_test_DNN, custom_loss_train_DNN, custom_loss_test_DNN, target_name, save_file_path, savename, scaler_X_NN, scaler_y_NN):
    
    y_actual_train_orig = scaler_y_NN.inverse_transform(y_actual_train)
    predicted_train_orig = scaler_y_NN.inverse_transform(predicted_train)

    results_train_dict = {
        'Actual Train Values from DNN': y_actual_train_orig.flatten(),
        'Predicted Train Values from DNN': predicted_train_orig.flatten(),
    }
    results_train_df = pd.DataFrame(results_train_dict)
    results_train_df.to_excel(os.path.join(save_file_path, f'{savename}_DNN_prediceted_train_data_{target_name}.xlsx'),index=False)

    y_actual_test_orig = scaler_y_NN.inverse_transform(y_actual_test)
    predicted_test_orig = scaler_y_NN.inverse_transform(predicted_test)

    results_test_dict = {
        'Actual Test Values from DNN': y_actual_test_orig.flatten(),
        'Predicted Test Values from DNN': predicted_test_orig.flatten()
    }
    results_test_df = pd.DataFrame(results_test_dict)
    results_test_df.to_excel(os.path.join(save_file_path, f'{savename}_DNN_prediceted_test_data_{target_name}.xlsx'),index=False)

################################################################
# Function to Save data for later optimization
################################################################

def save_model_with_metadata(model, scaler_X, scaler_y, save_file_path, savename, model_name, 
                             input_features, output_features, X_data, y_data):
    save_folder = os.path.join(save_file_path, savename)
    os.makedirs(save_folder, exist_ok=True)

    # Compute min/max directly from data
    X_min = X_data.min(axis=0).tolist()
    X_max = X_data.max(axis=0).tolist()
    y_min = y_data.min(axis=0).tolist() if y_data is not None else None
    y_max = y_data.max(axis=0).tolist() if y_data is not None else None

    # Save metadata
    metadata = {
        "model_name": model_name,
        "input_features": list(input_features),
        "output_features": list(output_features),
        "scaler_X_mean": scaler_X.mean_.tolist(),
        "scaler_X_scale": scaler_X.scale_.tolist(),
        "scaler_y_mean": scaler_y.mean_.tolist() if hasattr(scaler_y, "mean_") else None,
        "scaler_y_scale": scaler_y.scale_.tolist() if hasattr(scaler_y, "scale_") else None,
        "scaler_X_min": X_min,
        "scaler_X_max": X_max,
        "scaler_y_min": y_min,
        "scaler_y_max": y_max,
    }

    metadata_path = os.path.join(save_folder, f"{savename}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Model, scalers, and metadata saved at: {save_folder}")
    
def save_metadata_to_excel(save_file_path, savename, metadata_list):
    
    excel_path = os.path.join(save_file_path, savename, f"{savename}_model_info.xlsx")
    
    # Flatten metadata for Excel
    rows = []
    for md in metadata_list:
        rows.append({
            "Model Name": md["model_name"],
            "Input Features": ", ".join(md["input_features"]),
            "Output Features": ", ".join(md["output_features"]),
            "Scaler X Min": md["scaler_X_min"],
            "Scaler X Max": md["scaler_X_max"],
            "Scaler Y Min": md["scaler_y_min"],
            "Scaler Y Max": md["scaler_y_max"],
        })
    
    df = pd.DataFrame(rows)
    df.to_excel(excel_path, index=False)
    print(f"Metadata saved to Excel: {excel_path}")

################################################################
# Function for plotting Actual vs Prediction Graph for DNN
################################################################

def plot_results(elapsed_time_1, elapsed_time_2, elapsed_time_3, elapsed_time_total_DNN, 
                 X, y_actual_train, y_predicted_train, y_actual_test, y_predicted_test, 
                 huber_loss_train_DNN, huber_loss_test_DNN, rmse_train_DNN, rmse_test_DNN, 
                 quantile_loss_train_DNN, quantile_loss_test_DNN, R2_score_train_DNN, 
                 R2_score_test_DNN, custom_loss_train_DNN, custom_loss_test_DNN, 
                 scaler_X_NN, scaler_y_NN, target_name, savename):

    # Inverse transform
    y_actual_train_orig = scaler_y_NN.inverse_transform(y_actual_train)
    y_predicted_train_orig = scaler_y_NN.inverse_transform(y_predicted_train)
    y_actual_test_orig = scaler_y_NN.inverse_transform(y_actual_test)
    y_predicted_test_orig = scaler_y_NN.inverse_transform(y_predicted_test)

    # Dynamic plot sizing based on sample size
    n_train = y_actual_train.size
    n_test = y_actual_test.size
    total_layers = 1 + len(HIDDEN_LAYERS) + 1
    total_neurons = calculate_total_neurons(HIDDEN_LAYERS)
    num_iterations = n_train * NUM_EPOCHS

    # Dynamic figsize logic
   
    plt.figure(figsize=(8.5,5))

    # Adjust sizes dynamically
    total_layers_DNN = n_train + n_test
    min_scale = 0.5 
    max_scale = 2.5 
    base_size = 80
    scale_factor_DNN = np.clip(1000 / (total_layers_DNN + 1), min_scale, max_scale)
        
    # scale_factor_DNN = 1 / np.log10(total_layers_DNN + 10)
    s_size_DNN = 80 * scale_factor_DNN
    # Error metrics and annotations
    plt.annotate(fr'R$^2$ Score (Train): {R2_score_train_DNN:.5f}', (0.03, 0.73), fontsize=11.5, xycoords='axes fraction')
    plt.annotate(fr'R$^2$ Score (Test): {R2_score_test_DNN:.5f}', (0.03, 0.68), fontsize=11.5, xycoords='axes fraction')
    plt.annotate(fr'Quantile loss (Train): {quantile_loss_train_DNN:.2e}  (Test): {quantile_loss_test_DNN:.2e}', (0.03, 0.93), fontsize=11.5, xycoords='axes fraction')
    plt.annotate(fr'Huber loss (Train): {huber_loss_train_DNN:.2e}  (Test): {huber_loss_test_DNN:.2e}', (0.03, 0.88), fontsize=11.5, xycoords='axes fraction')
    plt.annotate(fr'RMSE (Train): {rmse_train_DNN:.2e}  (Test): {rmse_test_DNN:.2e}', (0.03, 0.83), fontsize=11.5, xycoords='axes fraction')
    plt.annotate(fr'$L_c$ (Train): {custom_loss_train_DNN:.2e} (Test): {custom_loss_test_DNN:.2e}', (0.03, 0.78), fontsize=11.5, xycoords='axes fraction')

    # Scatter plots
    plt.scatter(y_actual_train_orig.flatten(), y_predicted_train_orig.flatten(), color='royalblue', edgecolor='black', s = s_size_DNN,
                marker='o', label='Training Data')
    plt.scatter(y_actual_test_orig.flatten(), y_predicted_test_orig.flatten(), marker='^', s = s_size_DNN,
                label='Test Data', color='none', edgecolor='black')
    plt.plot([min(y), max(y)], [min(y), max(y)], color='black', alpha=0.8, linestyle='--', linewidth=1.7, label='Actual Values')

    # Box with model parameters
    box_props = dict(boxstyle='round,pad=0.4', edgecolor='black', facecolor='white', alpha=1)
    textstr = '\n'.join((
        'DNN Hyperparameters',
        '',
        fr'Input Dimension: {num_params}',
        fr'Total Samples: {n_test + n_train}',
        fr'Train Samples: {n_train}',
        fr'Test Samples: {n_test}',
        fr'Total NN Layers: {total_layers}',
        fr'Total Neurons: {total_neurons}',
        fr'Learning Rate: {LEARNING_RATE}',
        fr'No. of Iterations: {NUM_EPOCHS}',
        fr'Random State: {RANDOM_STATE}',
        fr'Training time: {elapsed_time_1:.3f} s',
        fr'Inference time (Train): {elapsed_time_2:.3f} s',
        fr'Inference time (Test): {elapsed_time_3:.3f} s',
        fr'Total time taken: {elapsed_time_total_DNN:.3f} s'
    ))
    plt.annotate(textstr, (1.03, 0.20), fontsize=10, xycoords='axes fraction', bbox=box_props, color="black")

    plt.xlabel(f'Actual Values ({y.name} [{Unit}])', fontsize=16)
    plt.ylabel(f'Predicted Values ({y.name} [{Unit}])', fontsize=16)
    #plt.title(f'{y.name} predictions from DNN with PReLU activation', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tick_params(axis='both', which='both', direction='in', labelsize=17, length=8,
               top=True, bottom=True, left=True, right=True, width=1.5)
    plt.tick_params(axis='both', which='minor', direction='in',
               length=4, width=1.5, top=True, bottom=True, left=True, right=True)
    ax = plt.gca()  # get current axes
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(AutoMinorLocator(10))
    ax.yaxis.set_minor_locator(AutoMinorLocator(7))
    
    plt.legend(loc='lower right', fontsize=12, facecolor='white', edgecolor='black', frameon=True, bbox_to_anchor=(0.97, 0.03))
    plt.grid(False)
    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(save_file_path, savename + '_DNN_Actual_vs_Prediceted_Data.jpg'), dpi=800)

################################################################
# Function to plot the errors with iterations with Adam in DNN
################################################################

def plot_loss(history, savename):
    loss_data = history.history['loss']
    val_loss_data = history.history.get('val_loss', [])
    
    final_train_loss = loss_data[-1]
    final_val_loss = val_loss_data[-1] if val_loss_data else None

    indices = range(0, len(loss_data), 3)
    points = int(NUM_EPOCHS * 0.05)

    indices_train = list(range(0, len(loss_data), points))
    indices_val = list(range(0, len(val_loss_data), points)) if val_loss_data else []

    plt.figure(figsize=(10, 4.5))

    if val_loss_data:
        plt.plot(
            range(len(val_loss_data)),
            val_loss_data,
            linestyle='-',
            color='darkgrey',
            label=f'Validation Loss (Final: {final_val_loss:.3e})'
        )
        plt.plot(
            indices_val,
            [val_loss_data[i] for i in indices_val],
            linestyle='',
            color='royalblue',
            marker='^',
            markersize=10,
            markerfacecolor='white',
            markeredgecolor='grey',
            markeredgewidth=2
        )

    plt.plot(
        indices,
        [loss_data[i] for i in indices],
        linestyle='-',
        color='royalblue',
        linewidth=1.5,
        marker='',
        markersize=8,
        markeredgecolor='purple',
        label=f'Training Loss (Final: {final_train_loss:.3e})'
    )

    plt.plot(
        indices_train,
        [loss_data[i] for i in indices_train],
        linestyle='',
        color='royalblue',
        marker='o',
        markersize=8,
        markerfacecolor='white',
        markeredgecolor='purple',
        markeredgewidth=2
    )

    plt.title(fr'DNN (PReLU) - Loss vs. Number of Epoch for ({y.name})', fontsize=20)
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.legend(fontsize=18, facecolor='white', edgecolor='black', frameon=True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tick_params(axis='both', which='both', direction='in', labelsize=17, length=8,
               top=True, bottom=True, left=True, right=True, width=1.5)
    plt.tick_params(axis='both', which='minor', direction='in',
               length=4, width=1.5, top=True, bottom=True, left=True, right=True)
    ax = plt.gca()  # get current axes
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(AutoMinorLocator(10))
    ax.yaxis.set_minor_locator(AutoMinorLocator(8))
    plt.grid(False)
    plt.tight_layout()

    # Saving the figure
    plt.savefig(os.path.join(save_file_path, savename + '_DNN_loss_vs_epoch.jpg'), dpi=800)
    # --- Save losses to Excel ---
    df = pd.DataFrame({
        "Epoch": list(range(1, len(loss_data) + 1)),
        "Training_Loss": loss_data,
        "Validation_Loss": val_loss_data if val_loss_data else [None] * len(loss_data)
    })
    
    excel_path = os.path.join(save_file_path, savename + '_DNN_loss_vs_epoch.xlsx')
    df.to_excel(excel_path, index=False)

    print(f"Saved training/validation loss values to {excel_path}")

################################################################
# Function to perform 5-fold CV with the DNN predicted results
################################################################
def plot_5fold_cv(model, X, y):

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    Lc_scores, r2_scores, rmse_scores = [], [], []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        Lc_scores.append(custom_loss(y_test, y_pred))
        r2_scores.append(r2_score(y_test, y_pred))
        rmse_scores.append(root_mean_squared_error(y_test, y_pred))

    # ---- Save results to Excel ----
    results_df = pd.DataFrame({
        "Fold": np.arange(1, 6),
        "R² Score": r2_scores,
        "Custom Loss (Lc)": Lc_scores,
        "RMSE": rmse_scores
    })
    results_df.to_excel(os.path.join(save_file_path, savename + '_DNN_5_fold_CV_results.xlsx'), index=True)

    # Plot
    folds = np.arange(1, 6)
    width = 0.25
    fig, ax1 = plt.subplots(figsize=(8, 5.5), constrained_layout=True)

    # Left axis: R² and Lc
    ax1.bar(folds - width, r2_scores, width,
            color='royalblue', edgecolor='black', label='R² Score')
    ax1.bar(folds, Lc_scores, width,
            color='darkgreen', edgecolor='black', label='Custom Loss')
    ax1.set_xlabel('Fold Number', fontsize=20)
    ax1.set_ylabel(fr'R$^2$ Score / $L_c$', fontsize=20)
    ax1.tick_params(axis='y', labelcolor='black')

    # Right axis: RMSE
    ax2 = ax1.twinx()
    ax2.bar(folds + width, rmse_scores, width,
            color='red', edgecolor='black', label='RMSE')
    ax2.set_ylabel('Root Mean Square Error', fontsize=20, color='black', labelpad=10)
    ax2.tick_params(axis='y', labelcolor='black')

    ax1.tick_params(axis='both', which='both', direction='in', labelsize=17,
                    length=8, top=True, bottom=True, left=True, right=False, width=1.5)
    ax1.tick_params(axis='both', which='minor', direction='in',
                    length=4, width=1.5, top=True, bottom=True, left=True, right=False)

    # Right axis (RMSE): ticks only on right
    ax2.tick_params(axis='both', which='both', direction='in', labelsize=17,
                    length=8, top=True, bottom=True, left=False, right=True, width=1.5)
    ax2.tick_params(axis='both', which='minor', direction='in',
                    length=4, width=1.5, top=True, bottom=True, left=False, right=True)

    # Minor ticks only for y-axes, no x-axis minor ticks
    ax1.yaxis.set_minor_locator(AutoMinorLocator(10))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(10))

    # Shared legend above the graph in 3 columns
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2
    fig.legend(handles, labels, loc='upper center',
               bbox_to_anchor=(0.5, 0.99), ncol=3,
               frameon=True, facecolor='white', edgecolor='black', fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(os.path.join(save_file_path, savename + '_DNN_5_fold_CV.jpg'), dpi=800)
    plt.show()

###################
# Model execution
###################
print(f"{BLUE_TEXT} Starting DNN...{RESET_TEXT}")
print("\n")

start_time_total_DNN = time.time() 

# Bulinding the deep neural network model.
model = build_model(X_train.shape[1])

# Compiling the deep neural network model.
history, elapsed_time_1 = compile_and_train(model, X_train, y_train, X_test, y_test)
     
# Calculating errors in the predicitons from DNN.
elapsed_time_2, elapsed_time_3, huber_loss_train_DNN, huber_loss_test_DNN, rmse_train_DNN, rmse_test_DNN, quantile_loss_train_DNN, quantile_loss_test_DNN, R2_score_train_DNN, R2_score_test_DNN, custom_loss_train_DNN, custom_loss_test_DNN, predicted_train, predicted_test = evaluate_model(model, X_test, y_test, X_train, y_train)    

end_time_total_DNN = time.time()
elapsed_time_total_DNN = end_time_total_DNN - start_time_total_DNN

# Saving the trained DNN model in '.h5' format for further use.
save_model(model,scaler_X_NN, scaler_y_NN, save_file_path, 'trained_model')
    
# Saving the converted Min-Max scaled values in '.joblib' format for futher use.
joblib.dump(scaler_X_NN, os.path.join(save_file_path, savename, savename + '_scaler_X.joblib'))
joblib.dump(scaler_y_NN, os.path.join(save_file_path, savename, savename + '_scaler_y.joblib'))
    
# Saving predictions from DNN in an excel file.
save_results_to_excel(y_train, predicted_train, y_test, predicted_test, huber_loss_train_DNN, huber_loss_test_DNN, rmse_train_DNN, rmse_test_DNN, quantile_loss_train_DNN, quantile_loss_test_DNN, R2_score_train_DNN, R2_score_test_DNN, custom_loss_train_DNN, custom_loss_test_DNN, 'Input_vs_flux', save_file_path, savename, scaler_X_NN, scaler_y_NN)

# Plotting the prediction from DNN and saving it in '.jpg' format.
plot_results(elapsed_time_1, elapsed_time_2, elapsed_time_3, elapsed_time_total_DNN, X_scaled, y_train, predicted_train, y_test, predicted_test, huber_loss_train_DNN, huber_loss_test_DNN, rmse_train_DNN, rmse_test_DNN, quantile_loss_train_DNN, quantile_loss_test_DNN, R2_score_train_DNN, R2_score_test_DNN, custom_loss_train_DNN, custom_loss_test_DNN, scaler_X_NN, scaler_y_NN, 'Input_vs_flux', savename)

# Perform 5-fold CV
plot_5fold_cv(model, X_scaled, y_scaled)

# Plot Loss vs. Epoch(Iteration) for DNN AdamW optimization in '.jpg' format.
plot_loss(history, savename)

# Saving model in json format
metadata_list = []

# Save model + scalers + metadata

save_model_with_metadata(
        model, 
        scaler_X_NN, 
        scaler_y_NN, 
        save_file_path,
        savename,
        model_name="DeepModel",
        input_features=X.columns, 
        output_features=[y.name],
        X_data=X.values, 
        y_data=y.values
        )    


# Load the saved metadata json for Excel export
with open(os.path.join(save_file_path, savename, f"{savename}_metadata.json")) as f:
    metadata = json.load(f)
metadata_list.append(metadata)

# Save all metadata to Excel
save_metadata_to_excel(save_file_path, savename, metadata_list)

print(f"{GREEN_TEXT}DNN training completed successfully!{RESET_TEXT}")

end_time_total = time.time()
elapsed_time_total = end_time_total - start_time_total

print(f"{BLUE_TEXT}\n Total runtime: {elapsed_time_total:.2f} seconds{RESET_TEXT}")
print(f"{GREEN_TEXT}\n Completed{RESET_TEXT}")
