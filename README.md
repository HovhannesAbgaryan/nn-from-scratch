
# Heart Disease Detection with Neural Networks

This project implements a pipeline for detecting heart disease using a neural network built from scratch in PyTorch. It includes data preprocessing, model training, and an API for real-time predictions.

---

## Project Structure

```
src/
├── nn_from_scratch/
│   ├── train/
│   │   ├── data_preprocessing.py
│   │   ├── models.py
│   │   ├── train.py
│   ├── api/
│   │   ├── load_nn.py
│   │   ├── model.py
├── train_model.py
├── app.py
```

### Key Components

1. **Data Preprocessing (`data_preprocessing.py`)**
    - Handles data loading, cleaning, and preprocessing.
    - Splits the data into training, validation, and testing sets.
    - Saves preprocessed data and preprocessing pipelines for reuse.

2. **Model Definition (`models.py`)**
    - Contains the neural network architecture (`HeartDiseaseDetector`) and building blocks (`Block`).
    - Supports flexible configurations such as batch normalization, dropout, and varying hidden layer sizes.

3. **Training Script (`train.py`)**
    - Automates the training and validation process.
    - Supports hyperparameter tuning for batch normalization, dropout, learning rate, and more.
    - Saves the best-performing model and training parameters.

4. **API (`app.py`)**
    - Provides an HTTP API for real-time predictions using FastAPI.
    - Includes endpoints for single and batch predictions.
    - Loads pretrained models and preprocessing pipelines.

5. **Model Loader (`load_nn.py`)**
    - Loads the saved model and preprocessing pipeline for inference.

6. **Pydantic Model (`model.py`)**
    - Defines a structured input schema for heart disease prediction features.
    - Provides validation for input data.

7. **Training Entry Point (`train_model.py`)**
    - Script to start the training process with specified data and model paths.

---

## Installation

### Prerequisites
- Python == 3.11
- [PyTorch](https://pytorch.org/)
- FastAPI
- Uvicorn
- Scikit-learn
- Joblib
- Pandas
- Numpy

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

### Data Preparation
Place the raw data files in `../data/archive/`. Ensure the dataset contains the following columns:

- **Numeric Columns**: `age`, `trestbps`, `chol`, `thalach`, `oldpeak`
- **Binary Columns**: `sex`, `fbs`, `exang`
- **Categorical Columns**: `cp`, `restecg`, `slope`, `ca`, `thal`
- **Target Column**: `target`

### Training the Model
Run the `train_model.py` script:
```bash
python src/train_model.py
```
This will:
- Preprocess the data.
- Train the model with hyperparameter tuning.
- Save the best model and preprocessing pipeline in the specified `model_path`.

### Running the API
Start the FastAPI application:
```bash
python src/app.py
```
The API will be available at `http://127.0.0.1:7676/`.

#### API Endpoints
1. **Single Prediction**
   - **Endpoint**: `/`
   - **Method**: POST
   - **Input**:
     ```json
     {
         "age": 45,
         "trestbps": 120,
         "chol": 230,
         "thalach": 150,
         "oldpeak": 1.5,
         "sex": 1,
         "fbs": 0,
         "exang": 0,
         "cp": 2,
         "restecg": 1,
         "slope": 1,
         "ca": 0,
         "thal": 2
     }
     ```
   - **Response**:
     ```json
     {
         "prediction": 0.85
     }
     ```

2. **Batch Prediction**
   - **Endpoint**: `/batch`
   - **Method**: POST
   - **Input**: List of objects with the same structure as single prediction.
   - **Response**:
     ```json
     {
         "predictions": [0.85, 0.45, 0.76]
     }
     ```

3. **Health Check**
   - **Endpoint**: `/`
   - **Method**: GET
   - **Response**:
     ```json
     {
         "version": "Heart-disease detector 0.1",
         "status": "OK"
     }
     ```

---

## Customization

### Hyperparameter Tuning
Modify the following parameters in `train.py` to experiment with different configurations:
- `hidden_size_variants`
- `use_batch_norm_variants`
- `use_dropout_variants`
- `dropout_rate_variants`
- `lr_variants`

### Neural Network Architecture
Update the `HeartDiseaseDetector` class in `models.py` to experiment with different architectures.
