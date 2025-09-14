import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, LSTM, Bidirectional, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import ctc_batch_cost

# Step 3: Upload and Preprocess Image
# Directly give your image path
image_path = r"C:\Users\chimm\Downloads\im3.png"  
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"❌ Image not found at: {image_path}")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.subplot(1, 2, 2)
plt.title("Preprocessed Image")
plt.imshow(thresholded, cmap="gray")
plt.axis("off")
plt.show()

# Step 4: OCR Using Tesseract
custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(thresholded, config=custom_config)
print("Recognized Text:")
print(text)

# Step 5: Define CNN + RNN Model
input_img = Input(shape=(128, 32, 1), name='image_input')
x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Reshape(target_shape=(-1, 128))(x)
x = Bidirectional(LSTM(128, return_sequences=True))(x)
x = Dense(128, activation='relu')(x)
output = Dense(len("abcdefghijklmnopqrstuvwxyz ") + 1, activation='softmax', name='output')(x)

model = Model(inputs=input_img, outputs=output)
model.compile(optimizer=Adam(), loss=ctc_batch_cost)
model.summary()

# Step 6: Generate Dummy Data for Training
X_train = np.random.rand(100, 128, 32, 1)
y_train = np.random.randint(1, 27, (100, 20))

# Train the model (Replace with actual dataset for better results)
model.fit(X_train, y_train, batch_size=10, epochs=2)

# Step 7: Test Model on Input Image
test_image = cv2.resize(thresholded, (128, 32)).reshape(1, 128, 32, 1) / 255.0
predictions = model.predict(test_image)
decoded_text = "".join([chr(np.argmax(char) + ord('a')) for char in predictions[0]])
print("Predicted Text:", decoded_text)

# Step 8: Save and Load the Model
model.save("handwriting_recognition_model.h5")
model = tf.keras.models.load_model("handwriting_recognition_model.h5", custom_objects={'ctc_batch_cost': ctc_batch_cost})
