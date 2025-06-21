# HOMLPT-NET

**Hierarchical Optimization-Driven Multi-Layer Perceptron with Adaptive Hyperparameter Tuning (HOMLPT-NET)**

HOMLPT-NET is a modular and extensible deep learning pipeline designed to facilitate efficient and automated regression modeling using deep neural networks. It integrates a user-friendly interface for data handling, preprocessing, model training, custom evaluation metrics, and intelligent file management. The system is structured to support rapid prototyping, automated tuning, and deployment with minimal manual intervention.

## Description

This framework incorporates:

- Intelligent file and dataset loading from local directories
- Automated preprocessing, including missing value checks and scaling
- Dynamic train-test split based on data availability
- Dynamic feature-target identification
- Custom loss and metric functions for regression tasks (with Huber, Quantile, RMSE, R², Composite Loss)
- Flexible data saving strategy for reproducibility and tracking
- Compatibility with Neural Architecture Search (NAS) for future extensions
- Support for both manual and GridSearchCV-based hyperparameter selection 
- Deep neural network with configurable hidden layers and PReLU activation
- Result saving (Excels, figures, scalers, model weights)
- Visual summaries of model accuracy and training behavior

---
## Key Modules and Functionality

### 1. Library Imports

The initial code block imports all necessary libraries including TensorFlow, Keras, NumPy, Pandas, Scikit-learn, Matplotlib, and others required for model development, evaluation, and interaction with the file system.

### 2. File Handling

The function `get_file_from_user()` prompts the user to specify the location of the dataset. Supported formats include:

- Comma-separated values (.csv)
- Microsoft Excel (.xls, .xlsx)
- Tab-separated text (.txt)

The system validates the file extension, confirms the file exists at the specified location, and loads it accordingly.
- Path validity and file format compatibility are verified.
- Data is read into a structured DataFrame.

### 3. Data Validation and Preprocessing

The `load_data()` function:

- Verifies the integrity of the dataset by checking for missing values.
- Offers the user the ability to drop unnecessary columns.
- Prompts for the identification of input (X) and output (y) columns.

After validation, the dataset is stored as three variables: `X_data`, `y_data`, and the full DataFrame `df`.

### 4. Column Unit Annotation

The user is asked to optionally specify the physical unit for the output column. This unit may be used later in plots or reports.

### 5. Model Feasibility Check

A validation step ensures that the number of features does not exceed the number of data samples. If the number of parameters is greater than the number of datapoints, the execution halts to prevent underfitting and numerical instability.

### 6. Automatic Train-Test Split Configuration

The system dynamically sets the proportion of data reserved for testing based on the total number of available datapoints:

- Less than 20 datapoints → 15 percent test data
- Between 20 and 29 datapoints → 20 percent test data
- Between 30 and 39 datapoints → 25 percent test data
- 40 datapoints or more → 30 percent test data

### 7. Output Directory and File Saving

The script requests a directory for saving results. Users can choose between their home directory or a custom path. A subdirectory named `DNN` is automatically created if it does not exist. The script then prompts for a filename to save the output in Excel format, validating for forbidden characters and potential overwrites.

### 8. Feature and Target Scaling

Two separate functions, `scale_data_x()` and `scale_data_y()`, normalize the feature set and output variable using `StandardScaler`. This ensures stable convergence and prevents dominance of any one feature due to scale disparity.

### 9. Data Splitting

The `split_data()` function divides the dataset into training and test sets using the predefined test size and a fixed random state for reproducibility.

### 10. Custom Loss and Evaluation Metrics

The system includes several domain-specific loss and evaluation metrics:

- `huber_loss`: A robust loss function less sensitive to outliers
- `quantile_loss`: Penalizes underestimates more heavily using a configurable quantile level (default is 0.7)
- `R2`: Computes the coefficient of determination
- `custom_loss`: A composite weighted metric that combines normalized versions of Huber Loss, RMSE, Quantile Loss, and R²

This composite score allows multi-objective optimization across various statistical indicators.

### 11. Utility Function

The function `calculate_total_neurons(layers)` computes the total number of neurons across a list of hidden layers. This aids in parameter tracking and model complexity assessment.

---

## 12. Hyperparameter Selection and Model Training

### User Input for Hyperparameters

The user selects either manual or automatic tuning.

---

### Manual Hyperparameter Entry

Users are prompted to provide:

- Learning rate
- Number of epochs
- Batch size
- Random state
- Hidden layers (minimum one, optional additional layers with descending neuron counts)

Input validation ensures valid entries and architectural consistency.

---

### Automatic Hyperparameter Tuning

Users provide the number of CPU processors. The script performs:

1. Wrapping `MLPRegressor` inside a `TrackedMLP` class
2. Defining a scoring dictionary with:
   - Huber Loss
   - Quantile Loss
   - RMSE
   - R²
   - Custom Loss
3. Defines a parameter grid with six blocks representing different architecture and hyperparameter combinations
(these blocks vary based on number of hidden layers, neurons per layer, learning rate, and batch size):

- Block 1 – Small network
(1 hidden layer, few neurons, small batch size, moderate learning rate)

- Block 2 – Medium-small network
(2 hidden layers, increasing neurons, slightly larger batch size)

- Block 3 – Medium network
(3 hidden layers, balanced neuron distribution, varied learning rates)

- Block 4 – Deep network
(3 hidden layers, higher neuron count in early layers, small batch size for stability)

- Block 5 – Wide-shallow network
(1–2 hidden layers, high neurons per layer, faster learning rates)

- Block 6 – Deep and wide network
(3 hidden layers, high neuron count, larger batch size, conservative learning rate)

These combinations are selected to explore a wide design space, balancing depth, width, and learning stability.

---

### Grid Search Execution

- Performed using `GridSearchCV` with 5-fold cross-validation
- The best model is selected based on the custom loss
- Models with excessive train-test performance gap are filtered out
- The best parameters are extracted and saved

---

### Model Training

Using the best hyperparameters:

- `MLPRegressor` is trained on the training dataset
- Predictions are made on both training and test sets
- Output values are inverse-transformed
- Performance is evaluated using:
  - Huber Loss
  - RMSE
  - Quantile Loss
  - R²
  - Custom Loss

Timing is tracked for training, inference, and total runtime.

---

### Output Files and Logs

The system saves:

- Grid Search results (complete and filtered)
- Best model’s loss curve
- Visual performance comparisons
- Optimal parameters and error metrics
- Actual vs predicted plots
- Time logs for training and inference

---

### Performance Visualization
---

### Actual vs Predicted Plot

A plot shows predicted vs actual values for both training and test datasets. Annotations include:

- R² scores
- All loss metrics
- Model architecture
- Execution time

### Metric Evolution Across Hyperparameter Combinations

Separate plots are generated for each metric:

- Test and train values
- Vertical and horizontal lines marking optimal values
- Annotated summary of best hyperparameters
- Automatically scaled figures based on model complexity

All figures are saved in high resolution for documentation.

---

## 13. DNN Model Construction

Based on the neural architecture selected through automated grid search or manual parameter input, the DNN model is then trained to enhance overall accuracy.

- Built using Keras `Sequential` API with:
  - Glorot initialization
  - PReLU activations
  - L2 regularization
- Output layer: Linear (regression)

---

### Training and Evaluation

- Model is compiled with AdamW optimizer.
- Training and inference times are logged.
- Evaluation metrics calculated on both train and test sets:
  - Huber loss
  - RMSE
  - Quantile loss
  - R² score
  - Custom loss

---

### Visualization and Reporting

#### Actual vs Predicted Plot

- Scatter plot showing training and testing predictions.
- Annotated with all metrics and model configuration.

#### Loss Curve

- Epoch-wise visualization of training and validation losses.
- Highlights convergence and potential overfitting.

---

### Output Saving

- Predictions saved in Excel files (`.xlsx`) for both training and testing.
- All plots saved as `.jpg`.
- Trained model saved in `.keras` format.
- Scalers saved as `.joblib`.


---

## Configuration Parameters

This section outlines all key configuration parameters used throughout the HOMLPT-NET pipeline. The parameters are grouped by their functional role for clarity.

---

### General Settings

| Parameter       | Description                                                        |
|----------------|--------------------------------------------------------------------|
| `RANDOM_STATE` | Integer used for reproducibility across different runs (default: 42) |
| `VERBOSE`      | Controls verbosity during model training (0 = silent, 1 = progress bar, 2 = one line per epoch) |
| `Unit`         | Optional string to annotate physical units in plots and metrics     |

---

### Data Processing and Preprocessing

| Parameter        | Description                                                       |
|------------------|-------------------------------------------------------------------|
| `INPUT_COLUMNS`  | List of input (feature) columns selected by the user              |
| `OUTPUT_COLUMNS` | List of output (target) column(s)                                 |
| `DROP_COLUMNS`   | Optional columns to be excluded from the analysis                 |
| `SCALE_INPUT`    | Whether to scale input features using StandardScaler              |
| `SCALE_OUTPUT`   | Whether to scale output using StandardScaler                      |
| `scaler_X_NN`    | Scikit-learn scaler object for input features                     |
| `scaler_y_NN`    | Scikit-learn scaler object for output values                      |

---

### Train-Test Split

| Parameter   | Description                                                                                         |
|-------------|-----------------------------------------------------------------------------------------------------|
| `TEST_SIZE` | Automatically determined based on dataset size:<br> - Less than 20 rows: 15%<br> - 20–29 rows: 20%<br> - 30–39 rows: 25%<br> - 40+ rows: 30% |

---

### Model Architecture

| Parameter            | Description                                                     |
|----------------------|-----------------------------------------------------------------|
| `HIDDEN_LAYERS`      | List of integers specifying neurons in each hidden layer        |
| `ACTIVATION`         | Activation function used in hidden layers (default: PReLU)      |
| `USE_BATCH_NORM`     | Boolean flag to include batch normalization                     |
| `USE_DROPOUT`        | Boolean flag to include dropout                                 |
| `DROPOUT_RATE`       | Dropout rate if dropout is enabled (e.g., 0.2)                  |
| `kernel_initializer` | Weight initializer (default: `glorot_uniform`)                  |
| `kernel_regularizer` | Regularizer applied to layer weights (default: `l2(0.0001)`)    |

---

### Training Parameters

| Parameter        | Description                                                              |
|------------------|--------------------------------------------------------------------------|
| `LEARNING_RATE`  | Learning rate for optimizer (e.g., 0.001)                                |
| `OPTIMIZER`      | Optimizer used (e.g., AdamW with weight decay)                          |
| `NUM_EPOCHS`     | Number of training iterations (default: 500)                             |
| `BATCH_SIZE`     | Size of batches for training (default: 16)                               |
| `EARLY_STOPPING` | Boolean flag to enable early stopping                                    |
| `PATIENCE`       | Number of epochs to wait for improvement before stopping (e.g., 40)      |
| `TOLERANCE`      | Minimum improvement threshold in monitored metric (e.g., 1e-12)          |

---

### Loss Functions and Evaluation Metrics

| Parameter             | Description                                                              |
|-----------------------|--------------------------------------------------------------------------|
| `LOSS_FUNCTION`       | Primary training loss (e.g., Mean Squared Error)                         |
| `CUSTOM_LOSS`         | Combined metric aggregating Huber, RMSE, Quantile Loss, and R²           |
| `QUANTILE_LEVEL`      | Alpha value used in quantile loss (default: 0.7)                         |
| `CUSTOM_LOSS_WEIGHTS` | Weights for combining metrics into a single custom loss                  |

---

### Hyperparameter Optimization

| Parameter         | Description                                                               |
|------------------|---------------------------------------------------------------------------|
| `USE_GRID_SEARCH` | Flag to activate GridSearchCV for hyperparameter tuning                  |
| `PARAM_GRID`      | Dictionary specifying the search space (neurons, learning rate, batch size, etc.) |
| `CV_FOLDS`        | Number of folds for cross-validation (default: 5)                         |
| `SCORING`         | Dictionary of scoring metrics (e.g., R², RMSE, Huber, Quantile, Custom)   |
| `MAX_PROCESSORS`  | Number of CPU cores used for parallel grid search                        |

---

### Visualization and Plotting

| Parameter          | Description                                                           |
|--------------------|-----------------------------------------------------------------------|
| `FIGURE_RESOLUTION`| Output resolution for plots (typically 300–400 DPI)                   |
| `PLOT_STYLE`       | Matplotlib style settings (e.g., `seaborn-darkgrid`, `ggplot`)        |
| `ANNOTATE_PLOTS`   | Whether to annotate evaluation metrics directly on plots              |
| `s_size_DNN`       | Scatter marker size, dynamically scaled to sample size                |
| `PLOT_SAVE_PATH`   | Directory where all generated plots will be saved                     |

---

### Output and File Management

| Parameter                 | Description                                                              |
|---------------------------|--------------------------------------------------------------------------|
| `save_file_path`          | Directory for saving all output files                                    |
| `savename`                | Filename prefix used consistently across saved files                     |
| `FILENAME`                | Output Excel filename for final results                                  |
| `EXPORT_MODEL`            | Whether to save trained model for future inference                       |
| `model_name`              | Model identifier (e.g., `trained_model`)                                 |
| `CREATE_SUBDIR_IF_MISSING`| Automatically creates subdirectories such as `/DNN` if absent            |
| `SAVE_FORMAT`             | Supported formats include `.xlsx`, `.csv`, `.keras`, `.joblib`           |
| `save_folder`             | Final directory path where model weights, scalers, and plots are stored  |

---

## Future Enhancements

- GPU-accelerated parallel training
- Advanced visualizations for convergence, residuals, and weight dynamics

---

## Software Requirements

- Python 3 or newer
- For specifics check the `Libraries to install` folder

---

## Author

This framework was developed as part of a research initiative focused on automated deep learning pipelines for regression modeling in tabular datasets.

---

## License

This project is available for academic and research use only. For commercial licensing or production use, please contact the author.
