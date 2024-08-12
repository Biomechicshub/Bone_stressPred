import os
file_path = "new.py"  # Use your specified path if applicable
if os.path.exists(file_path):
    print("File exists.")
else:
    print("File does not exist.")

import tensorflow as tf

# List GPUs and set memory growth here

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=5000, output_dim=16),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam')
print("Model created successfully!")


