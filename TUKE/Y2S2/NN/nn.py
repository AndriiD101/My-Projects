import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(9)

# Base Layer class
class Layer:
    def forward(self, inp):
        raise NotImplementedError

    def backward(self, inp, grad_outp):
        raise NotImplementedError

# ReLU activation layer
class ReLU(Layer):
    def forward(self, inp):
        self.inp = inp
        return np.maximum(0, inp)

    def backward(self, inp, grad_outp):
        return grad_outp * (self.inp > 0)

# Sigmoid activation layer
class Sigmoid(Layer):
    def forward(self, inp):
        self.inp = inp
        self.outp = 1 / (1 + np.exp(-inp))
        return self.outp

    def backward(self, inp, grad_outp):
        return grad_outp * self.outp * (1 - self.outp)

# Dense (fully-connected) layer
class Dense(Layer):
    def __init__(self, inp_units, outp_units, learning_rate=0.1):
        self.lr = learning_rate
        self.W = np.random.randn(inp_units, outp_units) * 0.1
        self.b = np.zeros(outp_units)

    def forward(self, inp):
        self.inp = inp
        return inp @ self.W + self.b

    def backward(self, inp, grad_outp):
        grad_W = inp.T @ grad_outp
        grad_b = np.sum(grad_outp, axis=0)
        grad_inp = grad_outp @ self.W.T

        # Update weights and biases
        self.W -= self.lr * grad_W
        self.b -= self.lr * grad_b
        return grad_inp

# Multi-Layer Perceptron class
class MLP:
    def __init__(self):
        self.layers = []

    def add_layer(self, neuron_count, inp_shape=None, activation='ReLU', learning_rate=0.1):
        if len(self.layers) == 0 and inp_shape is None:
            raise ValueError("inp_shape must be specified for the first layer")

        if len(self.layers) == 0:
            inp_units = inp_shape
        else:
            # Find the output size of the last Dense layer
            dense_layers = [layer for layer in self.layers if isinstance(layer, Dense)]
            inp_units = dense_layers[-1].W.shape[1] if dense_layers else inp_shape

        self.layers.append(Dense(inp_units, neuron_count, learning_rate))
        if activation is not None:
            if activation == 'ReLU':
                self.layers.append(ReLU())
            elif activation == 'Sigmoid':
                self.layers.append(Sigmoid())
            # For regression, passing activation=None produces a linear output

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def predict(self, X):
        return self.forward(X)

    # Modified fit method to record training and validation losses
    def fit(self, X, y, X_val=None, y_val=None, epochs=100):
        train_losses = []
        val_losses = []
        for epoch in range(epochs):
            # Forward pass on training data
            outp = self.forward(X)
            loss = np.mean((outp - y) ** 2)
            train_losses.append(loss)
            grad = 2 * (outp - y) / y.size

            # Backward pass through layers in reverse order
            for layer in reversed(self.layers):
                grad = layer.backward(layer.inp, grad)

            # Compute validation loss if validation data is provided
            if X_val is not None and y_val is not None:
                val_outp = self.forward(X_val)
                val_loss = np.mean((val_outp - y_val) ** 2)
                val_losses.append(val_loss)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
        return train_losses, val_losses

if __name__ == "__main__":
    # Load the California Housing dataset
    housing = fetch_california_housing()
    X, y = housing.data, housing.target.reshape(-1, 1)

    # Scale features and target for better training stability
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y)

    # Split the data into training and validation sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )

    # Build the MLP
    network = MLP()
    network.add_layer(64, inp_shape=X_train.shape[1], activation='ReLU', learning_rate=0.01)
    network.add_layer(32, activation='ReLU', learning_rate=0.01)
    # For regression, use a linear output layer (activation=None)
    network.add_layer(1, activation=None, learning_rate=0.01)

    # Train the network and record losses
    epochs = 200
    train_losses, val_losses = network.fit(X_train, y_train, X_val=X_test, y_val=y_test, epochs=epochs)

    # Evaluate on the validation set
    predictions = network.predict(X_test)
    test_loss = np.mean((predictions - y_test) ** 2)
    print("Test Loss:", test_loss)

    # -----------------------------------------
    # Graph 1: Visualizing the Dataset
    # Scatter plot of the first feature vs. target
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train[:, 0], y_train.ravel(), alpha=0.5)
    plt.xlabel("Feature 0 (scaled)")
    plt.ylabel("Target (scaled)")
    plt.title("Scatter Plot of Feature 0 vs Target")
    plt.show()

    # -----------------------------------------
    # Graph 2: Training and Validation Loss over Epochs
    plt.figure(figsize=(8, 6))
    epochs_range = range(1, epochs + 1)
    plt.plot(epochs_range, train_losses, label="Training Loss")
    plt.plot(epochs_range, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.show()
