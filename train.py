import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("🚀 Initiating NASA Kepler Data Sequence...")

# 1. Load the raw data
try:
    df = pd.read_csv('exoTrain.csv')
    print(f"✅ Data loaded successfully! Found {len(df)} star systems.")
except FileNotFoundError:
    print("❌ ERROR: Could not find 'exoTrain.csv'. Make sure it is in the exact same folder as this script.")
    exit()

# 2. Separate the answers (y) from the light curve data (X)
# The dataset uses 2 for planet, 1 for no planet. We subtract 1 to make it 1 and 0 (binary).
y = df['LABEL'] - 1 
X = df.drop('LABEL', axis=1)

# 3. Normalize the brightness values so the neural network doesn't get confused
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Reshape the 2D spreadsheet into a 3D tensor for the 1D Convolutional Neural Network
# Format: (Number of Stars, Time Steps, 1 Feature)
X_cnn = np.expand_dims(X_scaled, axis=2)

# 5. Split into 80% training data and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(X_cnn, y, test_size=0.2, random_state=42)

print(f"🪐 Training Data Shape: {X_train.shape}")
print(f"🧪 Testing Data Shape: {X_test.shape}")
print("Ready to build the Neural Network architecture!")
import tensorflow as tf
from tensorflow.keras import layers, models

print("🧠 Building the 1D Convolutional Neural Network...")

# 1. Define the Neural Network Architecture
model = models.Sequential([
    # The network acts like a scanner, looking for the "U-shape" dip across the 3,197 time steps
    layers.Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    layers.MaxPooling1D(pool_size=2),
    
    layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    
    # Compile the findings into a final decision
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2), # Randomly drops connections to prevent memorizing the data
    layers.Dense(1, activation='sigmoid') # Outputs a probability: close to 1 is a planet, close to 0 is not
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("⚙️ Commencing Training Sequence (this may take a few minutes)...")

# 2. Train the model using the 80% training data, and test it against the 20% unseen data
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# 3. Export the trained brain to use in the web app
model.save('exoplanet_model.h5')
print("💾 Model successfully trained and saved as 'exoplanet_model.h5'!")
import tensorflow as tf
from tensorflow.keras import layers, models

print("🧠 Building the 1D Convolutional Neural Network...")

# 1. Define the Neural Network Architecture
model = models.Sequential([
    # The network acts like a scanner, looking for the "U-shape" dip across the 3,197 time steps
    layers.Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    layers.MaxPooling1D(pool_size=2),
    
    layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    
    # Compile the findings into a final decision
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2), # Randomly drops connections to prevent memorizing the data
    layers.Dense(1, activation='sigmoid') # Outputs a probability: close to 1 is a planet, close to 0 is not
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("⚙️ Commencing Training Sequence (this may take a few minutes)...")

# 2. Train the model using the 80% training data, and test it against the 20% unseen data
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# 3. Export the trained brain to use in the web app
model.save('exoplanet_model.h5')
print("💾 Model successfully trained and saved as 'exoplanet_model.h5'!")
import tensorflow as tf
from tensorflow.keras import layers, models

print("🧠 Building the 1D Convolutional Neural Network...")

# 1. Define the Neural Network Architecture
model = models.Sequential([
    # The network acts like a scanner, looking for the "U-shape" dip across the 3,197 time steps
    layers.Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    layers.MaxPooling1D(pool_size=2),
    
    layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    
    # Compile the findings into a final decision
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2), # Randomly drops connections to prevent memorizing the data
    layers.Dense(1, activation='sigmoid') # Outputs a probability: close to 1 is a planet, close to 0 is not
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("⚙️ Commencing Training Sequence (this may take a few minutes)...")

# 2. Train the model using the 80% training data, and test it against the 20% unseen data
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# 3. Export the trained brain to use in the web app
model.save('exoplanet_model.h5')
print("💾 Model successfully trained and saved as 'exoplanet_model.h5'!")