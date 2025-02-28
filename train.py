import os
import tensorflow as tf
import tensorflow as tf
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


os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"


print(tf.__version__)


print(tf.test.is_built_with_cuda())  # Should print True
print(tf.test.is_gpu_available())    # Should print True

# Check if GPU is available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU is available.")
else:
    print("GPU is not available.")


logo = LeaveOneGroupOut()


# Load X and y here
X = np.load('segmented_data.npy')
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



def create_model(optimizer='adam', learning_rate=5.922703339963269e-05, dropout_rate=0.4216691331002174, lstm_units=204, dense_units= 259, activation='tanh'):
    input_shape = (100, 6)
    user_weights_input = tf.keras.layers.Input(shape=(1,))

    inputs = tf.keras.layers.Input(shape=input_shape)

    # Feature extractor
    # features = tf.keras.layers.Conv1D(dense_units, kernel, padding='same', activation=activation)(inputs)
    # features = tf.keras.layers.MaxPooling1D(2)(features)
    # features = tf.keras.layers.Conv1D(dense_units, kernel, padding='same', activation=activation)(features)
    # features = tf.keras.layers.MaxPooling1D(2)(features)
    features = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units, return_sequences=True))(inputs)
    features = tf.keras.layers.Dropout(dropout_rate)(features)
    features = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units, return_sequences=True))(features)
    features = tf.keras.layers.Dropout(dropout_rate)(features)
    # features = tf.keras.layers.Attention()([features, features])
    features = tf.keras.layers.Dense(dense_units, activation=activation)(features)
    features = tf.keras.layers.Dense(dense_units, activation=activation)(features)
    features = tf.keras.layers.Dense(dense_units, activation=activation)(features)
    features = tf.keras.layers.Flatten()(features)
    features = tf.keras.layers.concatenate([features, user_weights_input])
    # Main task (regression)
    regression_output = tf.keras.layers.Dense(24, activation='linear')(features)

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


weight = np.genfromtxt('bodyweight.csv', delimiter=',')
weights = np.repeat(weight, 3)


yy = y 
mape_values = []
all_preds = pd.DataFrame()
all_tests = pd.DataFrame()


for i, (train_index, test_index) in enumerate(logo.split(X, y, groups=users)):
    # mape_values = []  # Initialize an empty list to store MAPE valuesmape_values = []  # Initialize an empty list to store MAPE values
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    weights_train, weights_test = weights[train_index], weights[test_index]
    # Get the user groups for the training and test sets
    train_users = users[train_index]
    test_users = users[test_index]
    unique_train_users = np.unique(train_users)

    # X_train_bet,y_train_bet, X_test_bet, _, target_transformer33, tttt3, _ = normalize_data(
    #       X_train[users_bet],  y_train[users_bet], X_test, y_test)

    X_train, y_train, X_test, y_test_1, target_transformer, target_transformer2, input = normalize_data(X_train, y_train, X_test, y_test)

    print(y_test.mean())


    print('user ID: ', i)
    print(y_test.shape)

    # Prepare domain labels
    domain_labels_train = np.zeros((X_train.shape[0], 1))  # Source domain
    domain_labels_test = np.ones((X_test.shape[0], 1))     # Target domain


    # Create model
    model = create_model()

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
        epochs = 168,
        batch_size = 268,
        verbose=0
    )


    # Make predictions
    y_pred, _ = model.predict([X_test, weights_test])



    # Denormalize predictions separately using test set indices
    y_pred = target_transformer.inverse_transform(y_pred)
    y_pred=y_pred*weight[i]
    print(y_pred)
    # ... [rest of your evaluation code]
    y_pred = pd.DataFrame(y_pred)
    y_test = pd.DataFrame(y_test)
    yy = pd.DataFrame(yy)

    print(y_test)


    MAPE = np.mean(np.abs((pd.DataFrame((y_test.iloc[:,23:]).values)-pd.DataFrame((y_pred.iloc[:,23:]).values))/pd.DataFrame((y_test.iloc[:,23:]).values)))


    print(MAPE)
    mape_values.append(MAPE)  # Append the MAPE value to the list

    # Compute the mean of the MAPE values so far
    mean_mape = np.mean(mape_values)
    print("Mean MAPE after iteration", i, ":", mean_mape)
    

    all_preds = pd.concat([all_preds, y_pred], axis=0, ignore_index=True)
    all_tests = pd.concat([all_tests, y_test], axis=0, ignore_index=True)