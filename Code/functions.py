
import numpy as np
import pandas as pd

def perceptron(x, y, n, epochs=1000):
    """
    Train a Perceptron classifier using the Perceptron learning algorithm.

    Parameters:
    ----------
    x: The input feature matrix.
    y: The target labels that contain values -1 or 1.
    n: The learning rate for weight updates.
    epochs: The number of times to iterate over the entire dataset during training.

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
    x: The input feature matrix.
    y: The target labels that contain values -1 or 1.
    lam: The regularization parameter that controls the trade-off between maximizing the margin and minimizing the classification error.
    epochs: The number of times to iterate over the entire dataset during training.
    batch_size: The number of samples to process in each batch update step.

    Returns:
    -------
    w: The final weights of the SVM after training.
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


def regularized_logistic(x, y, lam, epochs, batch_size, initial_eta=0.01, decay_rate=0.01):
    """
    Train a Logistic Regression classifier using stochastic gradient descent.

    Parameters:
    ----------
    x: The input feature matrix with shape (n_samples, n_features).
    y: The target labels containing values 1 or -1.
    lam: The regularization parameter.
    epochs: The number of times to iterate over the entire dataset during training.
    batch_size : The number of samples to process in each batch update step.
    initial_eta: The initial learning rate.
    decay_rate: Decay rate for learning rate.

    Returns:
    -------
    w: The final weights of the logistic regression model after training.
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
       x: The input feature matrix with shape (n_samples, n_features).
       y: The target labels with shape (n_samples,).
       k: The number of folds for cross-validation.

       Returns:
       -------
       folds: A list of k tuples, where each tuple contains:
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
    x_train: The feature matrix
    y_train: The target labels
    x_val: The feature matrix
    y_val: The target labels
    model: A string indicating the model to be used ('perceptron', 'pegasos', 'logistic', 'kernel_perceptron', 'kernel_pegasos')
    params: A dictionary containing hyperparameters for the model. The keys depend on the model:
        - For 'perceptron': {'n': learning_rate, 'epochs': number_of_epochs}
        - For 'pegasos': {'lam': regularization_param, 'epochs': number_of_epochs, 'batch_size': batch_size}
        - For 'logistic': {'lam': regularization_param, 'epochs': number_of_epochs, 'batch_size': batch_size}
        - For 'kernel_perceptron': {'n': learning_rate, 'epochs': number_of_epochs, 'kernel': 'gaussian' or 'polynomial'
        , 'sigma': sigma, 'c': constant, 'd': degree}
        - For 'kernel_pegasos': {'lam': regularization_param, 'epochs': number_of_epochs, 'batch_size': batch_size,
          'kernel': 'gaussian' or 'polynomial', 'sigma': sigma, 'c': constant, 'd': degree}

    Returns:
    -------
    loss : The 0-1 loss of the model on the validation set.
    """
    if model == 'perceptron':
        w, _ = perceptron(x_train, y_train, params['n'], params['epochs'])
        y_val_pred = np.sign(np.dot(x_val, w))

    elif model == 'pegasos':
        w, _ = pegasos(x_train, y_train, params['lam'], params['epochs'], params['batch_size'])
        y_val_pred = np.sign(np.dot(x_val, w))

    elif model == 'logistic':
        w, _ = regularized_logistic(x_train, y_train, params['lam'], params['epochs'], params['batch_size'])
        y_val_pred = np.sign(np.dot(x_val, w))

    elif model == 'kernel_perceptron':
        if params['kernel'] == 'gaussian':
            alphas, _ = kernel_perceptron(x_train, y_train, params['n'], sigma=params['sigma'], epochs=params['epochs'],
                                          kernel='gaussian')
            k_val = kernel_matrix(x_val, x_train, sigma=params['sigma'], kernel='gaussian', mode='testing')

        elif params['kernel'] == 'polynomial':
            alphas, _ = kernel_perceptron(x_train, y_train, params['n'], c=params['c'], d=params['d'],
                                          epochs=params['epochs'], kernel='polynomial')
            k_val = kernel_matrix(x_val, x_train, c=params['c'], d=params['d'], kernel='polynomial', mode='testing')

        y_val_pred = np.sign(np.dot(k_val, alphas * y_train))

    elif model == 'kernel_pegasos':
        if params['kernel'] == 'gaussian':
            alphas, _ = kernel_pegasos(x_train, y_train, params['lam'], params['epochs'], params['batch_size'],
                                           kernel='gaussian', sigma=params['sigma'])
            k_val = kernel_matrix(x_val, x_train, sigma=params['sigma'], kernel='gaussian', mode='testing')

        elif params['kernel'] == 'polynomial':
            alphas, _ = kernel_pegasos(x_train, y_train, params['lam'], params['epochs'], params['batch_size'],
                                           kernel='polynomial', c=params['c'], d=params['d'])
            k_val = kernel_matrix(x_val, x_train, c=params['c'], d=params['d'], kernel='polynomial', mode='testing')

        y_val_pred = np.sign(np.dot(k_val, alphas * y_train))

    loss = zero_one_loss(y_val_pred, y_val, len(y_val))

    return loss


def generate_combinations(param_grid):
    """
    Generate all possible combinations of hyperparameters from the parameter grid.

    Parameters:
    ----------
    param_grid: A dictionary where keys are hyperparameter names and values are lists of possible values for those hyperparameters.

    Returns:
    -------
    combinations: A list of dictionaries, where each dictionary represents a combination of hyperparameters.
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
    model : A string indicating the model to be tuned ('perceptron', 'pegasos', 'logistic', 'kernel_perceptron', 'kernel_pegasos').
    param_grid : A dictionary where keys are hyperparameter names and values are lists of possible values for those hyperparameters.
    k: The number of folds for cross-validation (default is 10).

    Returns:
    -------
    best_params : A dictionary containing the best hyperparameters found during the grid search.
    best_loss : The lowest average 0-1 loss achieved during the grid search.
    """
    cv_folds = cross_validation_split(x, y, k)
    best_params = None
    best_loss = float('inf')  # Initialize with a very high loss

    param_combinations = generate_combinations(param_grid)

    for params in param_combinations:
        avg_loss = 0

        for i in range(k):
            # Extract the validation fold
            x_val, y_val = cv_folds[i]

            # Create the training set by combining the other folds
            x_train = np.vstack([cv_folds[j][0] for j in range(k) if j != i])
            y_train = np.hstack([cv_folds[j][1] for j in range(k) if j != i])

            # Evaluate the model
            loss = evaluate_model(x_train, y_train, x_val, y_val, model, params)
            avg_loss += loss

        # Calculate the average loss over all folds
        avg_loss /= k

        # Check if the current combination has the lowest loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_params = params

    return best_params, best_loss

def polynomial_feature_expansion(data):
    """
    Applies a polynomial feature expansion of degree 2 to the input data.

    Parameters:
    data: Input data where the last column is the label.

    Returns:
    expanded_data: Expanded version of the data with new columns including:
        - The squared terms of each feature
        - All pairwise products of the features
    """

    # Separate features and label
    x = data.iloc[:, :-1]  # All columns except the last one are features
    y = data.iloc[:, -1]  # The last column is the label

    # Number of original features
    n = x.shape[1]

    # Initialize lists to collect new feature columns
    feature_columns = []

    # Add original features
    feature_columns.extend(x.columns)

    # Add squared features and pairwise products
    for i in range(n):
        feature_columns.append(f'{x.columns[i]}^2')
        for j in range(i + 1, n):  # Avoid duplicate pairs
            feature_columns.append(f'{x.columns[i]}*{x.columns[j]}')

    # Initialize DataFrame for the expanded features
    expanded_features = pd.DataFrame(index=data.index, columns=feature_columns)

    # Fill in the original features
    expanded_features[x.columns] = x

    # Fill in the squared features and pairwise products
    k = len(x.columns)
    for i in range(n):
        expanded_features.iloc[:, k] = x.iloc[:, i] ** 2
        k += 1
        for j in range(i + 1, n):  # Avoid duplicate pairs
            expanded_features.iloc[:, k] = x.iloc[:, i] * x.iloc[:, j]
            k += 1

    # Convert all columns to float
    expanded_features = expanded_features.astype(float)

    # Combine the expanded features with the target variable
    expanded_data = pd.concat([expanded_features, y], axis=1)
    expanded_data.columns = list(expanded_features.columns) + [data.columns[-1]]  # Ensure proper column naming

    return expanded_data


def gaussian_kernel(x1, x2, sigma=1.0):
    """
    Compute the Gaussian kernel (Radial Basis Function kernel) between two vectors.

    Parameters:
    ----------
    x1: First input vector.
    x2: Second input vector.
    sigma: Standard deviation of the Gaussian distribution.

    Returns:
    -------
    kernel: The Gaussian kernel value between x1 and x2.
    """
    return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * sigma ** 2))

def polynomial_kernel(x1, x2, c=1.0, d=3):
    """
    Compute the Polynomial kernel between two vectors.

    Parameters:
    ----------
    x1: First input vector.
    x2: Second input vector.
    c: Constant term in the polynomial kernel.
    d: Degree of the polynomial.

    Returns:
    -------
    kernel: The Polynomial kernel value between x1 and x2.
    """
    return (np.dot(x1, x2) + c) ** d

def kernel_matrix(x1, x2=None, sigma=1, kernel='gaussian', c=1, d=2, mode='training'):
    """
    Compute the kernel matrix for the input data using either the Gaussian kernel or the Polynomial kernel.

    Parameters:
    ----------
    x1: The input feature matrix.
    x2: The input feature matrix (only needed for 'testing' mode).
    sigma: Standard deviation of the Gaussian distribution.
    kernel: The type of kernel to use ('gaussian' or 'polynomial').
    c: Constant term in the polynomial kernel.
    d: Degree of the polynomial.
    mode: Specifies the mode of computation:
        - 'training': Compute the kernel matrix for training data (x1 with itself).
        - 'testing': Compute the kernel matrix between test data (x1) and training data (x2).

    Returns:
    -------
    K: The kernel matrix.
    """
    if mode == 'training':
        num_samples = x1.shape[0]
        K = np.zeros((num_samples, num_samples))

        for i in range(num_samples):
            for j in range(i, num_samples):  # Compute only half of the matrix due to symmetry
                if kernel == 'gaussian':
                    K[i, j] = gaussian_kernel(x1[i], x1[j], sigma)
                elif kernel == 'polynomial':
                    K[i, j] = polynomial_kernel(x1[i], x1[j], c, d)
        K += K.T - np.diag(K.diagonal())  # Make the matrix symmetric

    elif mode == 'testing':
        if x2 is None:
            raise ValueError("x2 must be provided for testing mode.")
        num_samples_1 = x1.shape[0]
        num_samples_2 = x2.shape[0]
        K = np.zeros((num_samples_1, num_samples_2))

        for i in range(num_samples_1):
            for j in range(num_samples_2):
                if kernel == 'gaussian':
                    K[i, j] = gaussian_kernel(x1[i], x2[j], sigma)
                elif kernel == 'polynomial':
                    K[i, j] = polynomial_kernel(x1[i], x2[j], c, d)

    else:
        raise ValueError("Invalid mode. Choose 'training' or 'testing'.")

    return K

def kernel_perceptron(x, y, n, sigma=0.1, epochs=1, kernel='gaussian', c=1.0, d=3):
    """
    Train a Kernel Perceptron classifier using the specified kernel function.

    Parameters:
    ----------
    x: The input feature matrix.
    y: The target labels that contain values -1 or 1.
    n: The learning rate for weight updates.
    sigma: Standard deviation for the Gaussian kernel (used if kernel='gaussian').
    epochs: The number of times to iterate over the entire dataset during training.
    kernel: The type of kernel to use ('gaussian' or 'polynomial').
    c: Constant term in the polynomial kernel (used if kernel='polynomial').
    d: Degree of the polynomial (used if kernel='polynomial').

    Returns:
    -------
    alphas: The weights of the Perceptron in the kernel space.
    predictions: The predicted labels for the input data.
    """
    num_samples = x.shape[0]
    alphas = np.zeros(num_samples)  # Initialize weights in the dual space

    # Compute the kernel matrix using the specified kernel
    k = kernel_matrix(x, sigma=sigma, kernel=kernel, c=c, d=d, mode='training')

    for epoch in range(epochs):
        for i in range(num_samples):
            # Compute the prediction using the kernel matrix
            prediction = np.sum(alphas * y * k[i])

            # Perceptron update rule
            if y[i] * prediction <= 0:
                alphas[i] += n

    # Compute the predictions using the precomputed kernel matrix
    predictions = np.sign(np.dot(alphas * y, k))

    return alphas, predictions


def kernel_pegasos(x, y, lam, epochs, batch_size, kernel='gaussian', sigma=1.0, c=1.0, d=2):
    """
    Train a kernelized SVM classifier using the Pegasos algorithm with either Gaussian or Polynomial kernel.

    Parameters:
    ----------
    x: The input feature matrix.
    y: The target labels that contain values -1 or 1.
    lam: The regularization parameter.
    epochs: The number of epochs for training.
    batch_size: The number of samples per batch.
    kernel: The type of kernel ('gaussian' or 'polynomial').
    sigma: Standard deviation for the Gaussian kernel.
    c: Constant term for the polynomial kernel.
    d: Degree of the polynomial kernel.

    Returns:
    -------
    alphas: The final weights (dual variables).
    predictions: The predicted labels for the input data.
    """
    if batch_size > x.shape[0]:
        raise ValueError("Batch size cannot be larger than the number of samples in the dataset.")

    num_samples = x.shape[0]
    alphas = np.zeros(num_samples)
    t = 0

    # Compute the full kernel matrix
    k = kernel_matrix(x, kernel=kernel, sigma=sigma, c=c, d=d)

    for epoch in range(epochs):
        for _ in range(x.shape[0] // batch_size):
            t += 1
            eta = 1 / (lam * t)  # Update learning rate

            # Randomly sample a batch
            batch_indices = np.random.randint(0, x.shape[0], batch_size)

            # Select from the precomputed kernel matrix the batch rows and columns
            k_batch = k[batch_indices]

            for i in range(batch_size):
                # Decision function using the kernel
                decision_value = np.sum(alphas * y * k_batch[i]) * (1 / (lam * t))

                if y[batch_indices[i]] * decision_value < 1:
                    alphas[batch_indices[i]] += 1

    # Compute predictions on the training data
    predictions = np.sign(np.dot(k, alphas * y))

    return alphas, predictions






