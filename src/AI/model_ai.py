import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# 1. Model to analyze growth from images
def create_growth_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)  # Output is the growth value (e.g., plant height)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# 2. Model to predict water needs
def create_water_need_model():
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(3,)),  # Input: moisture, temperature, growth value
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Output: probability of needing water (0-1)
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 3. Function to train models
def train_growth_model(model, X_train, y_train, epochs=10):
    history = model.fit(X_train, y_train, epochs=epochs, validation_split=0.2)
    return history

def train_water_need_model(model, X_train, y_train, epochs=10):
    history = model.fit(X_train, y_train, epochs=epochs, validation_split=0.2)
    return history

# 4. Function to make predictions
def predict_growth(model, image):
    # Assume image is a numpy array of size (224, 224, 3)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return model.predict(image)[0][0]

def predict_water_need(model, moisture, temperature, growth):
    input_data = np.array([[moisture, temperature, growth]])
    return model.predict(input_data)[0][0]

# 5. Function to plot training results
def plot_growth_history(history):
    plt.plot(history.history['mae'], label='mean absolute error')
    plt.plot(history.history['val_mae'], label='validation mean absolute error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.show()

def plot_water_need_history(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    X_growth = np.random.rand(1000, 224, 224, 3)  # Simulated images
    y_growth = np.random.rand(1000, 1)  # Simulated growth values
    
    X_water = np.random.rand(1000, 3)  # Simulated moisture, temperature, and growth data
    y_water = np.random.randint(2, size=(1000, 1))  # Simulated water needs (0 or 1)

    # Create and train models
    growth_model = create_growth_model()
    water_need_model = create_water_need_model()

    growth_history = train_growth_model(growth_model, X_growth, y_growth)
    water_need_history = train_water_need_model(water_need_model, X_water, y_water)

    # Plot training results
    plot_growth_history(growth_history)
    plot_water_need_history(water_need_history)

    # Example predictions
    sample_image = np.random.rand(224, 224, 3)
    predicted_growth = predict_growth(growth_model, sample_image)
    print(f"Predicted growth: {predicted_growth}")

    moisture, temperature = 0.5, 25
    predicted_water_need = predict_water_need(water_need_model, moisture, temperature, predicted_growth)
    print(f"Predicted water need: {predicted_water_need}")
