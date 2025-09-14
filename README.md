📌 Handwriting Recognition using CNN + BiLSTM + CTC
Description

This project demonstrates a basic pipeline for handwriting recognition using both traditional OCR (Tesseract) and a deep learning model built with TensorFlow/Keras.

Steps Involved:

Upload and Preprocess Image:

User uploads a handwritten image.

Image is converted to grayscale and thresholded using Otsu’s method to improve clarity.

Preprocessed and original images are displayed side by side for comparison.

OCR with Tesseract:

The preprocessed image is passed to Tesseract OCR for initial text recognition.

This provides a quick baseline result.

Deep Learning Model (CNN + BiLSTM + CTC):

Input images are resized to (128, 32, 1).

A Convolutional Neural Network (CNN) extracts spatial features.

Features are reshaped and passed into a Bidirectional LSTM (BiLSTM) to capture sequential dependencies.

A Dense + Softmax layer predicts character probabilities.

CTC (Connectionist Temporal Classification) loss is used for alignment between predictions and labels.

Dummy Training:

For demonstration, random training data is generated (X_train, y_train).

The model is trained for 2 epochs just to showcase the training process (replace with a real dataset for meaningful results).

Prediction:

The uploaded image is resized and normalized before being passed into the trained model.

Predictions are decoded into text output.

Model Saving & Loading:

The trained model is saved as an .h5 file.

Later, it can be reloaded with custom CTC loss for further use.

Outcome

Provides a baseline OCR result using Tesseract.

Demonstrates a deep learning approach (CNN + BiLSTM + CTC) for handwriting recognition.

Can be extended with a proper handwritten dataset (e.g., IAM dataset) for accurate recognition.
