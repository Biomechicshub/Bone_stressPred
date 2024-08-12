import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

import tensorflow as tf
print(tf.__version__)

import tensorflow as tf
print(tf.test.is_built_with_cuda())  # Should print True
print(tf.test.is_gpu_available())    # Should print True

# Check if GPU is available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU is available.")
else:
    print("GPU is not available.")

import numpy as np; import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV,cross_val_score, GroupKFold,LeaveOneGroupOut,RandomizedSearchCV
# Simple k-fold cross-validation
from sklearn.model_selection import StratifiedKFold,KFold,GroupKFold
# from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
seed=42
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GroupKFold
import optuna


logo = LeaveOneGroupOut()


# Load X and y here
X = np.load('segmented_data.npy')
y = np.load('new_max_stress_data2.npy')
y = y[:, :]  # Shape is (150, 2)
X = np.concatenate([X[:, 2, :, 0:3], X[:, 2, :, 6:9]], axis=-1)  # Shape is (150, 100, 12)
# X = X[:, 2, :, 0:3]  # Shape is (150, 100, 12)


# assuming that each participant performed 3 trials
users = np.repeat(np.arange(0, 50), 3)  # this will create an array [1, 1, 1, 2, 2, 2, ..., 50, 50, 50]
trials = np.repeat(np.arange(1, 151), 1)  # this will create an array [1, 2, 3, 4, ..., 149, 150]

from sklearn import preprocessing

def normalize_data(X_train, y_train, X_test, y_test):
    # Get original shapes
    original_X_shape = X_train.shape
    original_y_shape = y_train.shape

    # Reshape the inputs to 2D
    X_train_reshaped = X_train.reshape(original_X_shape[0] * original_X_shape[1], -1)

    # Normalize the inputs using StandardScaler
    input_transformer = preprocessing.StandardScaler()
    X_train_normalized = input_transformer.fit_transform(X_train_reshaped)

    # Reshape the inputs back to the original shape
    X_train_normalized = X_train_normalized.reshape(original_X_shape)

    # Apply the same transformation to the test inputs
    X_test_reshaped = X_test.reshape(X_test.shape[0] * X_test.shape[1], -1)
    X_test_normalized = input_transformer.transform(X_test_reshaped)
    X_test_normalized = X_test_normalized.reshape(X_test.shape)

    # Reshape the targets to 2D
    y_train_reshaped = y_train.reshape(-1, original_y_shape[-1])
    y_test_reshaped = y_test.reshape(-1, original_y_shape[-1])

    # Normalize the targets using StandardScaler
    target_transformer = preprocessing.StandardScaler()
    y_train_normalized = target_transformer.fit_transform(y_train_reshaped)


    # Normalize y_test using the same target_transformer
    target_transformer2 = preprocessing.StandardScaler()
    y_test2_normalized = target_transformer2.fit_transform(y_test_reshaped)
    y_test_normalized = target_transformer.transform(y_test_reshaped)

    return X_train_normalized, y_train_normalized, X_test_normalized, y_test_normalized, target_transformer, target_transformer2, input_transformer

from tensorflow.keras import layers, models
from tcn import TCN
# lstm_units=64,
def create_advanced_model(optimizer='sgd', nb_filters = 64, kernel=3, dilations=[1, 2, 4], learning_rate=0.0004551727950524748, dropout_rate=0.24515812494395833, dense_units=100, activation='relu'):
    input_shape = (100, 6)
    user_weights_input = tf.keras.layers.Input(shape=(1,))

    inputs = tf.keras.layers.Input(shape=input_shape)

    # Feature extractor
    # features = tf.keras.layers.Conv1D(dense_units, kernel, padding='same', activation=activation)(inputs)
    # features = tf.keras.layers.MaxPooling1D(2)(features)
    # features = tf.keras.layers.Conv1D(dense_units, kernel, padding='same', activation=activation)(features)
    # features = tf.keras.layers.MaxPooling1D(2)(features)
    features = TCN(nb_filters=nb_filters, kernel_size=kernel, dilations=dilations, input_shape=input_shape, return_sequences=True)(inputs)

    # features = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units, return_sequences=True))(inputs)
    features = tf.keras.layers.Dropout(dropout_rate)(features)
    # features = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units, return_sequences=True))(features)
    # features = tf.keras.layers.Dropout(dropout_rate)(features)
    # features = tf.keras.layers.Attention()([features, features])
    # features = tf.keras.layers.Dense(dense_units, activation=activation)(features)
    # features = tf.keras.layers.Dense(dense_units, activation=activation)(features)
    # features = tf.keras.layers.Dense(dense_units, activation=activation)(features)
    features = tf.keras.layers.Flatten()(features)
    features = tf.keras.layers.concatenate([features, user_weights_input])
    # Main task (regression)
    regression_output = tf.keras.layers.Dense(14, activation='linear')(features)

    # Domain classifier (after gradient reversal)
    grl = GradientReversalLayer()(features)
    domain_classifier = tf.keras.layers.Dense(dense_units, activation=activation)(grl)
    domain_classifier = tf.keras.layers.Dropout(dropout_rate)(domain_classifier)
    # domain_classifier = tf.keras.layers.Dense(dense_units, activation=activation)(domain_classifier)
    domain_output = tf.keras.layers.Dense(1, activation='sigmoid')(domain_classifier)

    model = tf.keras.Model(inputs=[inputs, user_weights_input], outputs=[regression_output, domain_output])
    if optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer == 'adagrad':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError("Unsupported optimizer")
    model.compile(optimizer=optimizer, loss=['mse', 'binary_crossentropy'], loss_weights=[1, 0.5])

    return model

class GradientReversalLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x):
        return self.grad_reverse(x)

    def compute_output_shape(self, input_shape):
        return input_shape

    @staticmethod
    @tf.custom_gradient
    def grad_reverse(x):
        y = tf.identity(x)

        def custom_grad(dy):
            return -dy

        return y, custom_grad


weight = np.genfromtxt('bodyweight3.csv', delimiter=',')
weights = np.repeat(weight, 3)
yy=pd.DataFrame(y)
y_range =  yy.iloc[:, 2:].max()-yy.iloc[:,2:].min()


def objective(trial):
    # Define your hyperparameter search space
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop','adagrad'])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'sigmoid'])
    
    nb_filters = trial.suggest_int('nb_filters', 16, 128)
    num_layers = trial.suggest_int('num_layers', 2, 5)  # e.g., choosing between 2 to 6 layers
    dilations = [2**i for i in range(num_layers)]

    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
    # lstm_units = trial.suggest_int('lstm_units', 64, 512)
    dense_units = trial.suggest_int('dense_units', 50,300)
    kernel = trial.suggest_categorical('kernel', ['3','5','10','15'])
    epochs = trial.suggest_int('epochs', 50, 500)
    batch_size = trial.suggest_int('batch_size', 64, 512)
    
    # Initialize a list to store LOOCV results
    loocv_results = []
    mape_values = []


    for i, (train_index, test_index) in enumerate(logo.split(X, y, groups=users)):
        # mape_values = []  # Initialize an empty list to store MAPE valuesmape_values = []  # Initialize an empty list to store MAPE values
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        weights_train, weights_test = weights[train_index], weights[test_index]

        # Get the user groups for the training and test sets
        train_users = users[train_index]
        test_users = users[test_index]
        unique_train_users = np.unique(train_users)

        # Separate test data based on the condition
        # (1, 151)
        users_a = [user for user in [region_list]]
        users_b = [user for user in range(0, 50) if user not in users_a]
        # users_bet = [user for user in unique_train_users if y_train[train_users == user].mean() <= 58 and y_train[train_users == user].mean()> 45]
        print('above:', users_a)
        print('below:', users_b)
        # print('bet:', users_bet)

        X_train_a,y_train_b, X_test_above, _, target_transformer11, tttt1, _ = normalize_data(
            X_train[users_a],  y_train[users_a], X_test, y_test)
        X_train_b,y_train_b, X_test_below, _, target_transformer22, tttt2, _ = normalize_data(
            X_train[users_b],  y_train[users_b], X_test, y_test)
        # X_train_bet,y_train_bet, X_test_bet, _, target_transformer33, tttt3, _ = normalize_data(
        #       X_train[users_bet],  y_train[users_bet], X_test, y_test)
        # Combine the normalized data
        # X_train = np.vstack([X_train_above_50, X_train_below_50, X_train_bet])
        # y_train = np.vstack([y_train_above_50, y_train_below_50, y_train_bet])
        # X_test = np.vstack([X_test_above, X_test_below, X_test_bet])

        X_train, y_train, X_test, y_test_1, target_transformer,target_transformer2, input = normalize_data(X_train, y_train, X_test, y_test)

        print(y_test.mean())


        # if y_test.mean()>= 45:
        if i in users_b:
            print('user ID: ', i)
            print(y_test.shape)

            # Prepare domain labels
            domain_labels_train = np.zeros((X_train.shape[0], 1))  # Source domain
            domain_labels_test = np.ones((X_test.shape[0], 1))     # Target domain


            # Create model
            model = create_advanced_model()

            print(X_train.shape)
            print(y_train.shape)
            # Initial training on main task with placeholder domain labels
            model.fit(x=[X_train,weights_train],
                    y=[y_train, domain_labels_train],
                    epochs=50,
                    batch_size=200,
                    verbose=0)
            # Concatenate domain labels
            domain_labels = np.concatenate([domain_labels_train, domain_labels_test])

            # Concatenate X values (train and test)
            X_combined = np.concatenate([X_train, X_test], axis=0)  # if you want to concatenate along the rows
            weights_combined = np.concatenate([weights_train, weights_test], axis=0)


            # Placeholder for the regression task (since we don't want to train it here)
            regression_placeholder = np.zeros((X_combined.shape[0], y_train.shape[1]))


            model.fit(
                x = [X_combined,weights_combined],
                y = [regression_placeholder, domain_labels],
                epochs = epochs,
                batch_size = batch_size,
                verbose=0
            )


            # Make predictions
            y_pred, _ = model.predict([X_test, weights_test])

            # Denormalize predictions separately using test set indices
            y_pred = target_transformer11.inverse_transform(y_pred)
            y_pred=y_pred*weight[i]
            print(y_pred)
            # ... [rest of your evaluation code]
            y_pred = pd.DataFrame(y_pred)
            y_test = pd.DataFrame(y_test)
            print(y_test)


            MAE = mean_absolute_error (y_test.iloc[:,2:], y_pred.iloc[:,2:])
            MSE = mean_squared_error (y_test.iloc[:,2:], y_pred.iloc[:,2:])
            RMSE = np.sqrt(MSE)
            MAPE = np.mean(np.abs((pd.DataFrame((y_test.iloc[:,2:]).values)-pd.DataFrame((y_pred.iloc[:,2:]).values))/pd.DataFrame((y_test.iloc[:,2:]).values)))
            
            NRMSE = RMSE / y_range


###  note: reduce indentation and remove this part. 
            # print(MAE)
            # print(MSE)
            # print(RMSE)
            # print(MAPE)
            # mape_values.append(MAPE)  # Append the MAPE value to the list

            # # Compute the mean of the MAPE values so far
            # mean_mape = np.mean(mape_values)
            # print("Mean MAPE after iteration", i, ":", mean_mape)
            # mse = MSE
            # print(mse)
            # loocv_results.append(mse)
            # # Add pruning to stop unpromising trials early
            # if trial.should_prune():
            #     raise optuna.exceptions.TrialPruned()


        # elif y_test.mean() < 45:
        elif i in users_a:
            print('user ID: ', i)
            print( y_test.shape)

            # Prepare domain labels
            domain_labels_train = np.zeros((X_train.shape[0], 1))  # Source domain
            domain_labels_test = np.ones((X_test.shape[0], 1))     # Target domain

            # Create model
            model = create_advanced_model()

            print(X_train.shape)
            print(y_train.shape)
            # Initial training on main task with placeholder domain labels
            model.fit(x=[X_train,weights_train],
                    y=[y_train, domain_labels_train],
                    epochs=50,
                    batch_size=200,
                    verbose=0)
            # Concatenate domain labels
            domain_labels = np.concatenate([domain_labels_train, domain_labels_test])

            # Concatenate X values (train and test)
            X_combined = np.concatenate([X_train, X_test], axis=0)  # if you want to concatenate along the rows
            weights_combined = np.concatenate([weights_train, weights_test], axis=0)

            # Placeholder for the regression task (since we don't want to train it here)
            regression_placeholder = np.zeros((X_combined.shape[0], y_train.shape[1]))


            model.fit(
                x = [X_combined,weights_combined],
                y = [regression_placeholder, domain_labels],
                epochs = epochs,
                batch_size = batch_size,
                verbose=0
            )


            # Make predictions
            y_pred, _ = model.predict([X_test, weights_test])


            y_pred = target_transformer22.inverse_transform(y_pred)
            y_pred=y_pred * weight[i]
            print(y_pred)

            # ... [rest of your evaluation code]
            y_pred = pd.DataFrame(y_pred)
            y_test = pd.DataFrame(y_test)
            print(y_test)
            MAE = mean_absolute_error (y_test.iloc[:,2:], y_pred.iloc[:,2:])
            MSE = mean_squared_error (y_test.iloc[:,2:], y_pred.iloc[:,2:])
            RMSE = np.sqrt(MSE)
            MAPE = np.mean(np.abs((pd.DataFrame((y_test.iloc[:,2:]).values)-pd.DataFrame((y_pred.iloc[:,2:]).values))/pd.DataFrame((y_test.iloc[:,2:]).values)))

            NRMSE = RMSE / y_range
        
        print(MAE)
        print(MSE)
        print(RMSE)
        print(NRMSE)
        mape_values.append(NRMSE)  # Append the MAPE value to the list
        # mape_v2.append(NRMSE.loc[:,-5:])
        # mean_mape = np.mean(mape_v2)
        # Compute the mean of the MAPE values so far
        mean_mape = np.mean(mape_values)
        print("Mean NRMSE after iteration", i, ":", mean_mape)
        
        mse = MSE
        print(mse)
        loocv_results.append(mse)
        # Add pruning to stop unpromising trials early
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    # Calculate the mean of LOOCV results
        avg_loocv_mse = np.mean(loocv_results)
    return avg_loocv_mse



# Create a study object and optimize
study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=100)  # Adjust the number of trials as needed



# Print the best hyperparameters and value
print("Best Hyperparameters:", study.best_params)
print('Best performance: ', study.best_value)

import os

def save_results(trial, best_params, best_value):
    with open("filename", "a") as f:
        f.write(f"Trial #{trial.number}\n")
        f.write(f"Best Hyperparameters: {best_params}\n")
        f.write(f"Best performance (MSE): {best_value}\n")
        f.write("\n")


# Save results to a text file after each trial
for trial in study.trials:
    save_results(trial, trial.params, trial.value)
