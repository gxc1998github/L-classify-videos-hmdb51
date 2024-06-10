import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from utils.data_loader import load_data
from scripts.model_definition import create_model

# Load data
X, y = load_data()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model
input_shape = (None, 224, 224, 3)  # Update input_shape based on your data
model = create_model(input_shape)

# Training
checkpoint = ModelCheckpoint('outputs/models/best_model.h5', monitor='val_loss', save_best_only=True, mode='min')
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=4, callbacks=[checkpoint])

# Save the final model
model.save('outputs/models/final_model.h5')

print("Training completed and model saved.")