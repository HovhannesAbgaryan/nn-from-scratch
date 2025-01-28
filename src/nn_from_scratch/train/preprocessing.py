import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import joblib

mapping_columns = {
    "ST slope": "slope",
    "chest pain type": "cp",
    "cholesterol": "chol",
    "exercise angina": "exang",
    "fasting blood sugar": "fbs",
    "max heart rate": "thalach",
    "resting bps": "trestbps",
    "resting bp s": "trestbps",
    "resting ecg": "restecg"
}

# region Functions

# region Dataset Preprocessing

def preprocess_heart_disease_dataset(df_: pd.DataFrame) -> pd.DataFrame:
    df = df_.copy()
    df['cp'] = df['cp'] - 1
    df['slope'] = df['slope'].replace({0: np.nan})
    df['slope'] = df['slope'] - 1
    return df

def preprocess_cleveland_1(df_: pd.DataFrame) -> pd.DataFrame:
    df = df_.copy()
    df['cp'] = df['cp'] - 1
    df['slope'] = df['slope'].replace({0: np.nan})
    df['slope'] = df['slope'] - 1
    return df

def preprocess_cleveland_2(df_: pd.DataFrame) -> pd.DataFrame:
    df = df_.copy()
    df['ca'] = df['ca'].replace({4: np.nan})
    df['thal'] = df['thal'].replace({0: np.nan})
    df['thal'] = df['thal'] - 1
    return df

# endregion Dataset Preprocessing

def data_load_and_concatenation(data_path: str) -> pd.DataFrame:
    print("Datapath: ", data_path)

    if data_path.endswith(".csv"):
        df = pd.read_csv(data_path)

        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])

        df = df.rename(columns=mapping_columns)
        return df
    else:
        dfs = []
        for csv in os.listdir(data_path):
            print(csv)

            if not csv.endswith(".csv"):
                continue

            df = pd.read_csv(os.path.join(data_path, csv))

            if "Unnamed: 0" in df.columns:
                df = df.drop(columns=["Unnamed: 0"])

            df = df.rename(columns=mapping_columns)

            if "cleveland1" in csv:
                df = preprocess_cleveland_1(df)
            elif "cleveland2" in csv:
                df = preprocess_cleveland_2(df)
            elif "Dataset Heart Disease" in csv:
                df = preprocess_heart_disease_dataset(df)

            dfs.append(df)

        return pd.concat(dfs).drop_duplicates()


def data_split_and_preprocess(data_path: str, model_path: str) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], ColumnTransformer]:
    os.makedirs(model_path, exist_ok=True)

    # Sort the columns into numerical, binary and categorical types
    numeric_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    binary_columns = ['sex', 'fbs', 'exang']
    categorical_columns = ['cp', 'restecg', 'slope', 'ca', 'thal']

    all_columns = (numeric_columns + binary_columns + categorical_columns + ['target'])

    os.makedirs(os.path.join(data_path, "split"), exist_ok=True)

    # Check if data is split
    split_data_exists = (os.path.exists(os.path.join(data_path, 'split/train.csv')) and
                         os.path.exists(os.path.join(data_path, 'split/test.csv')) and
                         os.path.exists(os.path.join(data_path, "split/valid.csv")))

    if split_data_exists:
        print("Splits already exist. Reading the training, validation and testing datasets...")

        # Read training, validation and testing datasets
        train_data = pd.read_csv(os.path.join(data_path, 'split/train.csv'))[all_columns]
        valid_data = pd.read_csv(os.path.join(data_path, 'split/valid.csv'))[all_columns]
        test_data = pd.read_csv(os.path.join(data_path, 'split/test.csv'))[all_columns]

        X_train = train_data.drop("target", axis=1)
        y_train = train_data[['target']]

        X_valid = valid_data.drop("target", axis=1)
        y_valid = valid_data[['target']]

        X_test = test_data.drop("target", axis=1)
        y_test = test_data[['target']]
    else:
        print("Splits don't exist. Splitting the data into training, validation and testing datasets...")
        df = data_load_and_concatenation(data_path)[all_columns]

        # Split data into training and testing datasets
        X_train, X_test, y_train, y_test = train_test_split(df.drop("target", axis=1), df[['target']], test_size=0.3)

        # Split the testing dataset into validation and testing datasets
        X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5)

        # Save training dataset
        train_data = X_train.copy()
        train_data['target'] = y_train['target'].copy()
        train_data.to_csv(os.path.join(data_path, 'split/train.csv'), index=False)

        # Save validation dataset
        valid_data = X_valid.copy()
        valid_data['target'] = y_valid['target'].copy()
        valid_data.to_csv(os.path.join(data_path, 'split/valid.csv'), index=False)

        # Save testing dataset
        test_data = X_test.copy()
        test_data['target'] = y_test['target'].copy()
        test_data.to_csv(os.path.join(data_path, 'split/test.csv'), index=False)

    if os.path.exists(os.path.join(model_path, 'preprocess.joblib')) and split_data_exists:
        print("Loading preprocessing pipeline...")
        preprocessing: ColumnTransformer = joblib.load(os.path.join(model_path, 'preprocess.joblib'))
    else:
        print("No preprocessing pipeline, fitting and saving...")
        num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        cat_pipeline = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore"))
        bin_pipeline = make_pipeline(SimpleImputer(strategy="most_frequent"))
        preprocessing = ColumnTransformer([("num", num_pipeline, numeric_columns), ("cat", cat_pipeline, categorical_columns), ('bin', bin_pipeline, binary_columns)], remainder="drop")
        preprocessing.fit(X_train)
        joblib.dump(preprocessing, os.path.join(model_path, 'preprocess.joblib'))

    X_train_preprocessed = preprocessing.transform(X_train)
    X_valid_preprocess = preprocessing.transform(X_valid)
    X_test_preprocessed = preprocessing.transform(X_test)

    return (X_train_preprocessed, y_train.values), (X_valid_preprocess, y_valid.values), (X_test_preprocessed, y_test.values), preprocessing

# endregion Functions
