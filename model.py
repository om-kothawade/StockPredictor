import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models

# Load and preprocess the data
df = pd.read_csv('AAPL.csv')
df = df.drop(['Date'], axis='columns')

# Selecting the relevant columns for normalization
columns_to_normalize = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

# Initialize the scaler
scaler = MinMaxScaler()

# Normalize the data
normalized_data = scaler.fit_transform(df)

# Convert the normalized data back to a DataFrame for easier manipulation
normalized_df = pd.DataFrame(normalized_data, columns=columns_to_normalize)

# Display the first few rows of the normalized data
normalized_df.head()

# Define the sequence length
sequence_length = 10

# Prepare the sequences
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Create sequences from the normalized data
X_train, y_train = create_sequences(normalized_df.values, sequence_length)

# Display the shape of the training data
print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)

# Define the generator model using LSTM
def build_generator_lstm(input_dim=100, seq_length=10, feature_count=6):
    model = models.Sequential()
    model.add(layers.Dense(128, input_dim=input_dim))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.RepeatVector(seq_length))
    model.add(layers.LSTM(256, return_sequences=True))
    model.add(layers.LSTM(512, return_sequences=True))
    model.add(layers.TimeDistributed(layers.Dense(feature_count, activation='tanh')))
    return model

# Define the discriminator model using CNN
def build_discriminator_cnn(seq_length=10, feature_count=6):
    model = models.Sequential()
    model.add(layers.Conv1D(64, kernel_size=3, strides=1, input_shape=(seq_length, feature_count), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv1D(128, kernel_size=3, strides=1, padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.25))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

# Build the generator and discriminator
generator = build_generator_lstm()
discriminator = build_discriminator_cnn()

# Compile the discriminator
discriminator.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

# Build and compile the GAN
discriminator.trainable = False
gan_input = layers.Input(shape=(100,))
generated_sequence = generator(gan_input)
gan_output = discriminator(generated_sequence)
gan = models.Model(gan_input, gan_output)
gan.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), loss='binary_crossentropy')

# Display the models' summaries
generator.summary()
discriminator.summary()
gan.summary()

# Training parameters
epochs = 10000
batch_size = 32
half_batch = batch_size // 2

# Generate random noise for the generator's input
def generate_noise(batch_size, dim):
    return np.random.normal(0, 1, (batch_size, dim))

# Training loop
for epoch in range(epochs):
    # Train Discriminator
    idx = np.random.randint(0, X_train.shape[0], half_batch)
    real_sequences = X_train[idx]

    noise = generate_noise(half_batch, 100)
    fake_sequences = generator.predict(noise)

    d_loss_real = discriminator.train_on_batch(real_sequences, np.ones((half_batch, 1)))
    d_loss_fake = discriminator.train_on_batch(fake_sequences, np.zeros((half_batch, 1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train Generator
    noise = generate_noise(batch_size, 100)
    valid_y = np.array([1] * batch_size)
    g_loss = gan.train_on_batch(noise, valid_y)

    # Print the progress
    if epoch % 100 == 0:
        print(f"{epoch} [D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")

# Save the generator and discriminator models
generator.save('generator_model_lstm.h5')
discriminator.save('discriminator_model_cnn.h5')
