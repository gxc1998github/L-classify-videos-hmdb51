import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
from utils.data_loader import load_data

# Load data
X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load model
model = load_model('outputs/models/best_model.h5')

# Evaluate model
y_pred = model.predict(X_test)
y_pred = np.round(y_pred).astype(int)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy}")