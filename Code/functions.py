
import numpy as np
def perceptron(x, y, n, epochs=1000):
    """
    Train a Perceptron classifier using the Perceptron learning algorithm.

    Parameters:
    ----------
    x : The input feature matrix.
    y : The target labels that contain values -1 or 1.
    n : The learning rate for weight updates.
    epochs (optional): The number of times to iterate over the entire dataset during training.

    Returns:
    -------
    w = The final weights of the Perceptron after training, including the bias term.
    predictions = The predicted labels for the input data.
    """
    w = np.zeros(x.shape[1])  # Initialize weights to zero

    for epoch in range(epochs):
        for i in range(x.shape[0]):
            prediction = np.dot(w, x[i])
            # Perceptron update rule
            if prediction <= 0 and y[i] == 1:
                w = w + n * x[i]
            elif prediction > 0 and y[i] == -1:
                w = w - n * x[i]

    predictions = np.sign(np.dot(x, w))
    return w, predictions


def pegasos(x, y, lam, epochs, batch_size):
    """
    Train a Support Vector Machine (SVM) classifier using the Pegasos algorithm.

    Parameters:
    ----------
    x : The input feature matrix.
    y : The target labels that contain values -1 or 1.
    lam : The regularization parameter that controls the trade-off between maximizing the margin and minimizing the classification error.
    epochs : The number of times to iterate over the entire dataset during training.
    batch_size : The number of samples to process in each batch update step.

    Returns:
    -------
    w : The final weights of the SVM after training.
    predictions: The predicted labels for the input data.
    """

    if batch_size > x.shape[0]:
        return "Batch size cannot be larger than the number of samples in the dataset."

    # Initialize weight vector with zeros
    w = np.zeros(x.shape[1])
    t = 0  # Iteration counter
    w_sum = np.zeros_like(w)  # Initialize weight sum
    for epoch in range(epochs):
        for i in range(x.shape[0] // batch_size):
            t += 1
            # Learning rate as a function of the iteration count
            eta = 1 / (lam * t)
            # Randomly sample a batch of indices
            batch_indices = np.random.randint(0, x.shape[0], batch_size)
            x_batch = x[batch_indices]
            y_batch = y[batch_indices]

            # Update weights based on the current batch
            for j in range(batch_size):
                if y_batch[j] * np.dot(w, x_batch[j]) < 1:
                    w = (1 - eta * lam) * w + eta * y_batch[j] * x_batch[j]
                else:
                    w = (1 - eta * lam) * w

            w_sum += w

    w = w_sum/t
    predictions = np.sign(np.dot(x, w))

    return w, predictions


def split_data(data, train_ratio=0.7):
    """Split the dataset into training and testing sets."""
    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    x = np.insert(x, 0, 1, axis=1)  # Insert a column of ones for the bias term

    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    x = x[indices]
    y = y[indices]

    split_index = int(train_ratio * x.shape[0])

    x_train, x_test = x[:split_index], x[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    return x_train, x_test, y_train, y_test


def zero_one_loss(y_pred, y_true, number_of_samples):
    """
    Calculate the 0-1 loss.

    Parameters:
    y_true: True labels.
    y_pred: Predicted labels.
    number_of_samples:  Number of samples in the dataset.

    Returns:
    miscassifications: The 0-1 loss.
    """
    # Compute the number of misclassifications
    misclassifications = np.sum(y_pred != y_true) / number_of_samples

    return misclassifications