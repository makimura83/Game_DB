import os
import pandas as pd
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from datetime import datetime

# 1. Function to load images from folders (day-wise)
def load_images_by_day(base_folder):
    images_by_day = {}
    for folder in sorted(os.listdir(base_folder)):
        day_folder = os.path.join(base_folder, folder)
        if os.path.isdir(day_folder):
            images = []
            for file in os.listdir(day_folder):
                file_path = os.path.join(day_folder, file)
                image = load_img(file_path, target_size=(224, 224))
                image = img_to_array(image) / 255.0
                images.append(image)
            images_by_day[folder] = np.array(images)
    return images_by_day

# 2. Function to load CSV data (e.g., moisture, temperature, growth values)
def load_csv_data(csv_path):
    return pd.read_csv(csv_path)

# 3. Create model for plant growth prediction from images
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

# 4. Create model for predicting water needs based on moisture, temperature, and growth
def create_water_need_model():
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(3,)),  # Input: moisture, temperature, growth value
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Output: probability of needing water (0-1)
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 5. Function to train growth model
def train_growth_model(model, X_train, y_train, epochs=10):
    history = model.fit(X_train, y_train, epochs=epochs, validation_split=0.2)
    return history

# 6. Function to train water need model
def train_water_need_model(model, X_train, y_train, epochs=10):
    history = model.fit(X_train, y_train, epochs=epochs, validation_split=0.2)
    return history

# 7. Function to make predictions using the growth model
def predict_growth(model, image):
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return model.predict(image)[0][0]

# 8. Function to make predictions for water needs
def predict_water_need(model, moisture, temperature, growth):
    input_data = np.array([[moisture, temperature, growth]])
    return model.predict(input_data)[0][0]

# 9. Function to plot training results for growth model
def plot_growth_history(history):
    plt.plot(history.history['mae'], label='mean absolute error')
    plt.plot(history.history['val_mae'], label='validation mean absolute error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.show()

# 10. Function to plot training results for water need model
def plot_water_need_history(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# 11. Function to record watering events in a log file
def record_watering_event(log_file, plant_id, timestamp, moisture_level):
    with open(log_file, 'a') as file:
        file.write(f"{plant_id},{timestamp},{moisture_level}\n")

# Example usage
if __name__ == "__main__":
    # Paths to images and CSV file
    base_folder = 'path_to_image_folders'
    csv_path = 'path_to_csv'
    
    # Load images and CSV data
    images_by_day = load_images_by_day(base_folder)
    csv_data = load_csv_data(csv_path)
    
    # Prepare the data (this step will vary based on your CSV structure)
    X_growth = np.random.rand(1000, 224, 224, 3)  # Simulated image data
    y_growth = csv_data['growth'].values  # Simulated growth values
    
    X_water = csv_data[['moisture', 'temperature', 'growth']].values
    y_water = csv_data['need_water'].values  # 0 or 1 for water need
    
    # Create and train models
    growth_model = create_growth_model()
    water_need_model = create_water_need_model()
    
    growth_history = train_growth_model(growth_model, X_growth, y_growth)
    water_need_history = train_water_need_model(water_need_model, X_water, y_water)
    
    # Plot training results
    plot_growth_history(growth_history)
    plot_water_need_history(water_need_history)
    
    # Example predictions
    sample_image = np.random.rand(224, 224, 3)  # Example image
    predicted_growth = predict_growth(growth_model, sample_image)
    print(f"Predicted growth: {predicted_growth}")
    
    moisture, temperature = 0.5, 25  # Example inputs
    predicted_water_need = predict_water_need(water_need_model, moisture, temperature, predicted_growth)
    print(f"Predicted water need: {predicted_water_need}")
    
    # Record a watering event if the water need is high
    if predicted_water_need > 0.5:  # Threshold for watering
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record_watering_event('watering_log.csv', 'plant_1', timestamp, moisture)
        print(f"Watering event recorded at {timestamp}")

