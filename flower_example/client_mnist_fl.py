import flwr as fl
import tensorflow as tf
from tensorflow.keras import layers, models  # 建立CNN架構
import numpy as np  # 資料前處理

'''
Step 1. Build Local Model (建立本地模型)
'''
# Hyperparameter超參數
num_classes = 10
input_shape = (28, 28, 1)

# Build Model
model = models.Sequential([
    tf.keras.Input(shape=input_shape),
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax"),
])
# model.summary()

# Defines the loss function, the optimizer and the metrics
model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])

'''
Step 2. Load local dataset (引入本地端資料集)
'''
# Load data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Data preprocessing
# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

'''
Step 3. Get the configuration (weights of global model, hyperparameters of training and evaluating) from the server-side. (繼承NumPyClient類別，定義匯入模型權重、模型訓練及評估等的函式，其中的權重值、訓練及評估的超參數皆來自Server端)
'''


class MnistClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=3,
                  batch_size=256)  # steps_per_epoch=3
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}


'''
Step 4. Create an instance of our New-NumPyClient and add one line to actually run this client. (使用一行指令，同時產生NumPyClient類別的實例(物件)以執行模型的訓練及評估，並建立Client-to-Server的連線)
'''
fl.client.start_numpy_client(
    server_address="127.0.0.1:8080", client=MnistClient())
