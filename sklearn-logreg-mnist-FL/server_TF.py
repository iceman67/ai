import flwr as fl
import tensorflow as tf

# Load model for server-side parameter initialization
model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

# Get model weights as a list of NumPy ndarray's
weights = model.get_weights()

# Serialize ndarrays to `Parameters`
parameters = fl.common.ndarrays_to_parameters(weights)

# Use the serialized parameters as the initial global parameters
strategy = fl.server.strategy.FedAvg(
    initial_parameters=parameters,
)
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3), 
    strategy=strategy)