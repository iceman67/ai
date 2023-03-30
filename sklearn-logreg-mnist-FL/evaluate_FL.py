import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score

import pickle

import argparse
import utils

# Load MNIST dataset from https://www.openml.org/d/554
(X_train, y_train), (X_test, y_test) = utils.load_mnist()

def evaluate_models(batch, cid):
    
    for i in range (1, batch+1):
        # loading LogisticRegression Model
        fname = f'./model/client_{cid}-{i}-minist.pkl'
        with open(fname, 'rb') as f:
            model = pickle.load(f)
        loss = log_loss(y_test, model.predict_proba(X_test))
        y_predict = model.predict(X_test) 
        accuracy = accuracy_score(y_test, y_predict)
        print ( f'{i} : {loss},  "accuracy": {accuracy}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--batch",
        type=int,
        default=5,
        help=f"number of batch(default 5))",
    )
    parser.add_argument(
        "--cid", type=str, required=True, help="Client CID (no default)")
     
    args = parser.parse_args()
    evaluate_models(batch=args.batch, cid=args.cid)