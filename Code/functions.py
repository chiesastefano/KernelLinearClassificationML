import numpy as np

def perceptron(x, y, n):
    w = np.zeros(x.shape[1] + 1)    # Initialize the weights + bias term
    x = np.insert(x, 0, 1, axis=1)  # Add the bias term to the input data
    for i in range(x.shape[0]):
        prediction = np.dot(w, x[i])
        # Perceptron update rule
        if prediction <= 0 and y[i] == 1:
            w = w + n * x[i]
        elif prediction > 0 and y[i] == -1:
            w = w - n * x[i]
    return w



