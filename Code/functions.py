
import numpy as np
def perceptron(x, y, n, epochs=1):
    """
    Train a Perceptron classifier using the Perceptron learning algorithm.

    Parameters:
    ----------
    x : The input feature matrix
    y : The target labels that contain values -1 or 1.
    n : The learning rate for weight updates.
    epochs (optional): The number of times to iterate over the entire dataset during training.

    Returns:
    -------
    w = The final weights of the Perceptron after training, including the bias term.
    """
    x = np.insert(x, 0, 1, axis=1)  # Insert a column of ones for the bias term
    w = np.zeros(x.shape[1])  # Initialize weights to zero

    for epoch in range(epochs):
        for i in range(x.shape[0]):
            prediction = np.dot(w, x[i])
            # Perceptron update rule
            if prediction <= 0 and y[i] == 1:
                w = w + n * x[i]
            elif prediction > 0 and y[i] == -1:
                w = w - n * x[i]

    return w


