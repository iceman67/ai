# Flower Example using scikit-learn

This example of Flower uses `scikit-learn`'s `LogisticRegression` model to train a federated learning system. It will help you understand how to adapt Flower for use with `scikit-learn`.
Running this example in itself is quite easy.

We use two clients in a testing device environment:
```
-- Jetson nano
-- Raspberry PI 4
```
## Prerequisite 

We first need to install Flower, scikit-learn and openml as follows:
```

$ pip install flwr
$ pip install scikit-learn
$ pip install openml

```
## Project Setup

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/sklearn-logreg-mnist . && rm -rf flower && cd sklearn-logreg-mnist
```

This will create a new directory called `sklearn-logreg-mnist` containing the following files:

```shell
-- pyproject.toml
-- client.py
-- server.py
-- utils.py
-- README.md
```

Project dependencies (such as `scikit-learn` and `flwr`) are defined in `pyproject.toml`. 


# Run Federated Learning with scikit-learn and Flower

Afterwards you are ready to start the Flower server as well as the clients. You can simply start the server in a terminal as follows:

```shell
python server.py
```

Now you are ready to start the Flower clients which will participate in the learning. To do so simply open two more terminals and run the following command in each:

```shell
python client.py --cid A --server_address 192.168.1.116:8080
python client.py --cid B --server_address 192.168.1.116:8080
```
where each client gives two arguments `cid` and `server_address` 
You will see that Flower is starting a federated training. 

You can use the following command to evaluate the models that built by client

```shell
 python evaluate_FL.py --batch 5 --cid B
```
