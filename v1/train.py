import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
import pickle

# 1. Point to your Parquet file
FILE_PATH = "MSSQL-training.parquet"  

print(f"[*] Loading CICDDoS2019 Parquet file: {FILE_PATH}...")
# read_parquet handles compressed files directly
df = pd.read_parquet(FILE_PATH)

# Clean up column names just in case there are hidden spaces
df.columns = df.columns.str.strip()

print("[*] Cleaning data and handling infinite/NaN values...")
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# 2. Select exactly 16 critical numerical features to form a 4x4 matrix
# (Note: Double-check your column names if the script errors out here, 
# as some parquet versions use lowercase or slightly different names)
features_to_use = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Bwd Packet Length Max',
    'Bwd Packet Length Min', 'Flow Bytes/s', 'Flow Packets/s', 
    'Fwd Packets/s', 'Bwd Packets/s', 'URG Flag Count', 
    'Down/Up Ratio', 'Average Packet Size'
]

# Ensure features exist in this specific parquet file
features_to_use = [f for f in features_to_use if f in df.columns]
if len(features_to_use) < 16:
    print(f"[!] Warning: Found only {len(features_to_use)} matching features. padding or using alternative selection...")
    # Dynamically grab the first 16 numerical columns if predefined list doesn't match perfectly
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features_to_use = [c for c in numerical_cols if c not in ['Label', 'Class']][:16]

print(f"[*] Selected 16 features for 4x4 grid: {features_to_use}")

X = df[features_to_use].values

# Convert text labels into numerical 0 (Benign) and 1 (Attack)
# Real Parquet data might use 'Label' or 'Class'
label_col = 'Label' if 'Label' in df.columns else 'Class'
y = df[label_col].apply(lambda x: 0 if 'BENIGN' in str(x).upper() else 1).values

print(f"[+] Cleaned Dataset Shape: {X.shape}")
print(f"[+] Class distribution -> Benign (0): {np.sum(y==0)} | Attack (1): {np.sum(y==1)}")

# 3. Normalize features safely between 0 and 1
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

with open("scaler_cic2019.pkl", "wb") as f:
    pickle.dump(scaler, f)

# RESHAPE: Map the 16 features directly into a completely filled 4x4 matrix grid
X_images = X_scaled.reshape(-1, 4, 4, 1)

# Split data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X_images, y, test_size=0.2, random_state=42, stratify=y)

# 4. Advanced 2D CNN Architecture
model = models.Sequential([
    layers.Conv2D(32, (2, 2), activation='relu', input_shape=(4, 4, 1), padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (2, 2), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

print("[*] Launching training loop over parquet layers...")
model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=128,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping]
)

model.save("cic2019_2d_cnn.h5")
print("\n[====>] SUCCESS: Realistic 2D Model saved as 'cic2019_2d_cnn.h5'")

y_pred = (model.predict(X_test) > 0.5).astype(int)
print("\n" + "="*20 + " REAL WORLD PERFORMANCE REPORT " + "="*20)
print(classification_report(y_test, y_pred, target_names=["Benign Users", "DDoS Attackers"]))
