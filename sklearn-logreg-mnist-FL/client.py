import warnings
import flwr as fl
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import pickle

import argparse
import utils
import os
import timeit
import csv

DEFAULT_SERVER_ADDRESS = "[::]:8080"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--server_address",
        type=str,
        default=DEFAULT_SERVER_ADDRESS,
        help=f"gRPC server address (default: {DEFAULT_SERVER_ADDRESS})",
    )
    parser.add_argument(
        "--cid", type=str, required=True, help="Client CID (no default)")
     
    args = parser.parse_args()

    cwd = os.getcwd()

    if not os.path.exists(f"{cwd}/model"):
        os.makedirs(f"{cwd}/model")


    # Load MNIST dataset from https://www.openml.org/d/554
    (X_train, y_train), (X_test, y_test) = utils.load_mnist()

    # Split train set into 10 partitions and randomly use one for training.
    partition_id = np.random.choice(10)
    (X_train, y_train) = utils.partition(X_train, y_train, 10)[partition_id]

    # Create LogisticRegression Model
    model = LogisticRegression(
        penalty="l2",
        max_iter=1,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting
    )
    
    # Setting initial parameters, akin to model.compile for keras models
    utils.set_initial_params(model, param=None)
    
    # Define Flower client
    class MnistClient(fl.client.NumPyClient):

        def __init__(self, cid: str):
            self.cid = cid
            fid = open(f"{cwd}/{cid}-report.csv", 'w', newline="")
            self.writer = csv.writer(fid) 
            header = ['cid', 'server_round', 'elapsed']
            self.writer.writerow(header)

        def get_parameters(self, config):  # type: ignore
            param = utils.get_model_parameters(model)
            print (f'keys = {model.state_dict().keys()}')
            print (param[1])
            return utils.get_model_parameters(model)
        
        # NA
        def model_keys(self, model):
            print (model.state_dict().keys())
            #params_dict = zip(model.state_dict().keys(), parameters)
            

        # set the local model weights
        # train the local model
        # receive the updated local model weights
        def fit(self, parameters, config):  # type: ignore

            if config['server_round'] == None:
                server_round = 'None'
            else:
                server_round = config['server_round']

            utils.set_model_params(model, parameters)
            fit_begin = timeit.default_timer()

            # Ignore convergence failure due to low local epochs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)

            fit_duration = timeit.default_timer() - fit_begin
            pickle.dump(model, open(f"model/client_{args.cid}-{server_round}-minist.pkl",'wb'))

            print(f"Training finished for round {config['server_round']}, fitting time {fit_duration}")
            self.writer.writerow([self.cid,config['server_round'], fit_duration])
            return utils.get_model_parameters(model), len(X_train), {}
        
        # test the local model
        def evaluate(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            param = utils.get_model_parameters(model)
            
            '''
            print (model.get_params().keys())
            params_dict = zip(model.get_params().keys(), parameters)
            for k, v in params_dict:
                print (k, v)
            '''
           
            loss = log_loss(y_test, model.predict_proba(X_test))
            print (f'loss = {loss}')
            accuracy = model.score(X_test, y_test)
            return loss, len(X_test), {"accuracy": accuracy}

    # Start Flower client
    print(f'Starting client {args.cid}')
    print(f'server address : {args.server_address}')
    
    fl.client.start_numpy_client(server_address=args.server_address, client=MnistClient(args.cid))
