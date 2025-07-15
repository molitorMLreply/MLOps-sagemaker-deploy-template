import argparse
import os
import json
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import pickle

def model_fn(model_dir):
    model_path = os.path.join(model_dir, "model.pkl")
    return pickle.load(model_path)

def parse_args():
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=0.1)
    parser.add_argument('--model-type', type=str, required=True)
    # an alternative way to load hyperparameters via SM_HPS environment variable.
    parser.add_argument('--sm-hps', type=json.loads, default=os.environ['SM_HPS'])

    # input data and model directories
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    return parser.parse_known_args()

def load_data(path):
    df = pd.read_csv(os.path.join(path, "train.csv"))
    y = df.iloc[:,0]
    X = df.iloc[:, 1:]
    return X, y

def main():
    args, _ = parse_args()
    X, y = load_data(args.train)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
    learning_rate = args.learning_rate
    model_type = args.model_type
    if model_type == "xgboost":
        model = GradientBoostingRegressor(learning_rate=learning_rate)
    elif model_type == "randomforest":
        model = RandomForestRegressor()
    elif model_type == "mlp":
        model = MLPRegressor(learning_rate_init=learning_rate)
    model.fit(X = X_train, y = y_train)
    value = mean_squared_error(y_true = y_test, y_pred = model.predict(X_test))
    print(f"validation:rmse={value}")
    with open(os.path.join(args.model_dir, "model.pkl"), "wb") as file:
        pickle.dump(model, file)

if __name__ =='__main__':
    main()

