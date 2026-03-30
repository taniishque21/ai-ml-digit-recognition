import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# STEP 3: Load dataset
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# STEP 4: Show one image
plt.imshow(x_train[0], cmap='gray')
plt.title(f"Label: {y_train[0]}")
plt.show()

# STEP 5: Normalize data
x_train = x_train / 255.0
x_test = x_test / 255.0

# STEP 6: Build model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# STEP 7: Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# STEP 8: Train model
model.fit(x_train, y_train, epochs=5)

# Save model
model.save("digit_model.h5")
print("Model saved successfully!")

# STEP 9: Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_acc)

# STEP 10: Make prediction
predictions = model.predict(x_test)

print("Predicted:", np.argmax(predictions[0]))
print("Actual:", y_test[0])

# BONUS: Show prediction
plt.imshow(x_test[0], cmap='gray')
plt.title(f"Predicted: {np.argmax(predictions[0])}")
plt.show()