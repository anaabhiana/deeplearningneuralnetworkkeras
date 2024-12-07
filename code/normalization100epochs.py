# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load the data
url = "https://cocl.us/concrete_data"
data = pd.read_csv(url)

# Separate predictors and target
X = data.drop(columns=['Strength'])
y = data['Strength']

# Function to build the model
def build_model():
    model = Sequential([
        Dense(10, activation='relu', input_shape=(X.shape[1],)),  # Hidden layer with 10 nodes
        Dense(1)  # Output layer
    ])
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
    return model

# List to store mean squared errors
mse_list = []

# Repeat the process 50 times
for i in range(50):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
    
    # Normalize the data manually
    X_train_mean = X_train.mean()
    X_train_std = X_train.std()
    X_train_normalized = (X_train - X_train_mean) / X_train_std
    X_test_normalized = (X_test - X_train_mean) / X_train_std
    
    # Build and train the model
    model = build_model()
    model.fit(X_train_normalized, y_train, epochs=100, batch_size=10, verbose=0)  # Increased epochs to 100
    
    # Evaluate the model
    y_pred = model.predict(X_test_normalized)
    mse = mean_squared_error(y_test, y_pred)
    mse_list.append(mse)

# Calculate the mean and standard deviation of the mean squared errors
mean_mse_100_epochs = np.mean(mse_list)
std_mse_100_epochs = np.std(mse_list)

print(f"Mean of Mean Squared Errors (100 Epochs): {mean_mse_100_epochs}")
print(f"Standard Deviation of Mean Squared Errors (100 Epochs): {std_mse_100_epochs}")