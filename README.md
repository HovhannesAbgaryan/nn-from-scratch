
# Heart Disease Risk Prediction with Neural Networks

This project implements a pipeline for predicting heart disease risk using a neural network built from scratch in PyTorch on [Heart Disease Dataset](https://www.kaggle.com/datasets/hosammhmdali/heart-disease-dataset). It includes data preprocessing, model training, and an API for real-time predictions.

---

## Project Structure

```
src/
├── nn_from_scratch/
│   ├── api/
│   │   ├── load_nn.py
│   │   ├── model.py
│   ├── train/
│   │   ├── models.py
│   │   ├── preprocessing.py
│   │   ├── train.py
├── app.py
├── model_train.py
```

### Key Components

1. **Data Preprocessing (`preprocessing.py`)**
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

7. **Training Entry Point (`model_train.py`)**
    - Script to start the training process with specified data and model paths.

---

## Installation

### Prerequisites
- [Python](https://www.python.org/) == 3.11
- [Numpy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [PyTorch](https://pytorch.org/)
- [mypy](https://mypy-lang.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Pydantic](https://docs.pydantic.dev/latest/)
- [Joblib](https://joblib.readthedocs.io/en/stable/)
- [Uvicorn](https://www.uvicorn.org/)

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

### Data Preparation
Place the raw data files in `data`. Ensure the dataset contains the following columns:

- **Numeric Columns**: `age`, `trestbps`, `chol`, `thalach`, `oldpeak`
- **Binary Columns**: `sex`, `fbs`, `exang`
- **Categorical Columns**: `cp`, `restecg`, `slope`, `ca`, `thal`
- **Target Column**: `target`

For more information about columns, refer [here](https://archive.ics.uci.edu/dataset/45/heart+disease).

### Training the Model
Run the `model_train.py` script:
```bash
cd src
python model_train.py
```
This will:
- Preprocess the data.
- Train the model with hyperparameter tuning.
- Save the best model and preprocessing pipeline in the specified `model_path`.

### Running the API
Start the FastAPI application:
```bash
cd src
python app.py
```
The API will be available at `http://127.0.0.1:7878/`.

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
