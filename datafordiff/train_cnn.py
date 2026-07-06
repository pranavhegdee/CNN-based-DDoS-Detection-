import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# 1. Load Generated Dataset
print("[*] Loading dataset...")
df = pd.read_csv("dataset_ddos.csv")

# Extract features and categorical labels
X_raw = df.drop(columns=["label"]).values
y_raw = df["label"].values

# 2. Encode Labels (Map strings to integers 0-7)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw)
num_classes = len(label_encoder.classes_)

print(f"[*] Map configuration identified: {dict(zip(label_encoder.classes_, range(num_classes)))}")

# 3. Scale Features (MinMax Standardizer)
print("[*] Normalizing feature matrices...")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_raw)

# Shape enforcement: Ensure we have exactly 64 features for an 8x8 matrix
if X_scaled.shape[1] < 64:
    print(f"[!] Warning: Data has {X_scaled.shape[1]} features. Padding to 64 columns...")
    padding = np.zeros((X_scaled.shape[0], 64 - X_scaled.shape[1]))
    X_scaled = np.hstack((X_scaled, padding))
elif X_scaled.shape[1] > 64:
    print(f"[*] Truncating features from {X_scaled.shape[1]} to top 64 channels...")
    X_scaled = X_scaled[:, :64]

# Reshape data into an 8x8 grid with 1 channel (Grayscale Image Format)
X_images = X_scaled.reshape(-1, 8, 8, 1)

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_images, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 5. Build Deep 2D CNN Architecture
print("[*] Compiling deep learning network model...")
model = models.Sequential([
    layers.Input(shape=(8, 8, 1)),
    
    # First Convolutional Block
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    
    # Second Convolutional Block
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    
    # Flattening and Dense Classification Layers
    layers.Flatten(),  # Converts output matrix to vector shapes
    layers.Dense(1024, activation='relu'),  # Satisfies dense matrix dependencies
    layers.Dropout(0.4),
    layers.Dense(128, activation='relu'),
    
    # Softmax output layer handles probabilities for all 8 categories
    layers.Dense(num_classes, activation='softmax')
])

# Use Sparse Categorical Crossentropy so we don't have to manually One-Hot encode integer labels
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 6. Callbacks for Optimal Generalization (Prevents Overfitting)
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)

# 7. Run Training Execution Window
print("[*] Initiating model optimization training...")
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=1
)

# 8. Evaluate Operational Accuracy Matrix
print("\n[*] Validating performance metrics on holdout data...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"🥇 Target Test Categorization Accuracy: {test_acc * 100:.2f}%")

# 9. Save Assets for `cnn_guard.py` Implementation
print("[*] Exporting model architecture files...")
model.save("traffic_cnn_model_premium.h5")

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("[+] Done! All model pipeline components successfully stored.")
