
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
        raise ValueError("Batch size cannot be larger than the number of samples in the dataset.")

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


import numpy as np


def regularized_logistic(x, y, lam, epochs, batch_size, initial_eta=0.01, decay_rate=0.01):
    """
    Train a Logistic Regression classifier using stochastic gradient descent.

    Parameters:
    ----------
    x : The input feature matrix with shape (n_samples, n_features).
    y : The target labels containing values 1 or -1.
    lam : The regularization parameter.
    epochs : The number of times to iterate over the entire dataset during training.
    batch_size : The number of samples to process in each batch update step.
    initial_eta : The initial learning rate.
    decay_rate : Decay rate for learning rate.

    Returns:
    -------
    w : The final weights of the logistic regression model after training.
    predictions: The predicted labels for the input data.
    """

    if batch_size > x.shape[0]:
        raise ValueError("Batch size cannot be larger than the number of samples in the dataset.")

    # Convert labels from 1/-1 to 0/1 for logistic regression
    y_transformed = (y + 1) / 2  # Transform 1 -> 1 and -1 -> 0

    # Initialize weight vector with small random values
    w = np.random.randn(x.shape[1])
    t = 0  # Iteration counter

    for epoch in range(epochs):
        # Shuffle the data at the beginning of each epoch
        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)

        for i in range(0, x.shape[0], batch_size):
            t += 1
            # Learning rate with decay
            eta = initial_eta / (1 + decay_rate * t)

            # Randomly sample a batch of indices
            batch_indices = indices[i:i + batch_size]
            x_batch = x[batch_indices]
            y_batch = y_transformed[batch_indices]

            # Compute the predictions (probabilities) using the sigmoid function
            logits = np.dot(x_batch, w)
            logits = np.clip(logits, -709, 709)
            prediction = 1 / (1 + np.exp(-logits))

            # Compute the gradient of the logistic loss
            errors = prediction - y_batch
            gradient = np.dot(x_batch.T, errors) / batch_size + lam * w

            # Update weights
            w -= eta * gradient

    # Make predictions on the full dataset
    final_logits = np.dot(x, w)
    final_logits = np.clip(final_logits, -709, 709)
    prediction_prob = 1 / (1 + np.exp(-final_logits))

    # Convert probabilities to class labels (0 and 1), then transform back to -1 and 1
    prediction_transformed = (prediction_prob > 0.5).astype(int)
    prediction = 2 * prediction_transformed - 1  # Transform 0 -> -1 and 1 -> 1

    return w, prediction


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


def cross_validation_split(x, y, k):
    """
       Split the dataset into k folds for cross-validation.

       Parameters:
       ----------
       x : The input feature matrix with shape (n_samples, n_features).
       y : The target labels with shape (n_samples,).
       k : The number of folds for cross-validation.

       Returns:
       -------
       folds : A list of k tuples, where each tuple contains:
           - x_fold : The feature matrix for the current fold (validation set).
           - y_fold : The target labels for the current fold (validation set).
       """
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    x, y = x[indices], y[indices]

    folds = []
    fold_size = len(y) // k

    for i in range(k):
        if i == k - 1:
            fold_indices = indices[i * fold_size:]
        else:
            fold_indices = indices[i * fold_size: (i + 1) * fold_size]

        folds.append((x[fold_indices], y[fold_indices]))

    return folds


def evaluate_model(x_train, y_train, x_val, y_val, model, params):
    """
    Train and evaluate a model using the given parameters and data.

    Parameters:
    ----------
    x_train : The feature matrix for the training set
    y_train : The target labels for the training set
    x_val : The feature matrix for the validation set
    y_val : The target labels for the validation set
    model : A string indicating the model to be used ('perceptron', 'pegasos', 'logistic').
    params : A dictionary containing hyperparameters for the model. The keys depend on the model:
        - For 'perceptron': {'n': learning_rate, 'epochs': number_of_epochs}
        - For 'pegasos': {'lam': regularization_param, 'epochs': number_of_epochs, 'batch_size': batch_size}
        - For 'logistic': {'lam': regularization_param, 'epochs': number_of_epochs, 'batch_size': batch_size}

    Returns:
    -------
    loss : The 0-1 loss of the model on the validation set.
    """
    if model == 'perceptron':
        w, _ = perceptron(x_train, y_train, params['n'], params['epochs'])
    elif model == 'pegasos':
        w, _ = pegasos(x_train, y_train, params['lam'], params['epochs'], params['batch_size'])
    elif model == 'logistic':
        w, _ = regularized_logistic(x_train, y_train, params['lam'], params['epochs'], params['batch_size'])

    y_val_pred = np.sign(np.dot(x_val, w))
    loss = zero_one_loss(y_val_pred, y_val, len(y_val))

    return loss


def generate_combinations(param_grid):
    """
    Generate all possible combinations of hyperparameters from the parameter grid.

    Parameters:
    ----------
    param_grid : A dictionary where keys are hyperparameter names and values are lists of possible values for those hyperparameters.

    Returns:
    -------
    combinations : A list of dictionaries, where each dictionary represents a combination of hyperparameters.
    """
    keys = list(param_grid.keys())
    combinations = []

    def backtrack(index, current_combination):
        if index == len(keys):
            combinations.append(current_combination.copy())
            return
        key = keys[index]
        for value in param_grid[key]:
            current_combination[key] = value
            backtrack(index + 1, current_combination)

    backtrack(0, {})
    return combinations


def grid_search(x, y, model, param_grid, k=5):
    """
    Perform a grid search to find the best hyperparameters for a model using k-fold cross-validation.

    Parameters:
    ----------
    x : The input feature matrix
    y : The target labels
    model : A string indicating the model to be tuned ('perceptron', 'pegasos', 'logistic').
    param_grid : A dictionary where keys are hyperparameter names and values are lists of possible values for those hyperparameters.
    k : The number of folds for cross-validation (default is 10).

    Returns:
    -------
    best_params : A dictionary containing the best hyperparameters found during the grid search.
    best_loss : The lowest average 0-1 loss achieved during the grid search.
    """
    cv_folds = cross_validation_split(x, y, k)
    best_params = None
    best_loss = float('inf') # any first loss will be better than this

    param_combinations = generate_combinations(param_grid)

    for params in param_combinations:
        avg_loss = 0

        for fold in cv_folds:
            x_val, y_val = fold
            # Check if the train set is not equal to the validation set
            x_train = np.vstack([f[0] for f in cv_folds if not np.array_equal(f[0], x_val)])
            y_train = np.hstack([f[1] for f in cv_folds if not np.array_equal(f[1], y_val)])

            loss = evaluate_model(x_train, y_train, x_val, y_val, model, params)
            avg_loss += loss

        avg_loss /= k

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_params = params

    return best_params, best_loss













