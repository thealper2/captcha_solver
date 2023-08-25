import os
import cv2
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import save_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def train(dataset_dir, width, height, epochs):
	characters = os.listdir(dataset_dir)
	num_classes = len(characters)
	char_to_label = {char: label for label, char in enumerate(characters)}

	data = []
	labels = []

	for char in characters:
	    char_folder = os.path.join(dataset_dir, char)
	    char_images = os.listdir(char_folder)

	    for char_image in char_images:
	        image_path = os.path.join(char_folder, char_image)
	        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
	        image = cv2.resize(image, (width, height))
	        data.append(image)
	        labels.append(char_to_label[char])

	data = np.array(data, dtype=np.float32) / 255.0
	labels = np.array(labels)

	train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

	model = models.Sequential([
	    Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, 1)),
	    MaxPooling2D((2, 2)),
	    Flatten(),
	    Dense(128, activation='relu'),
	    Dense(num_classes, activation='softmax')
	])

	model.compile(optimizer='adam',
	              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	              metrics=['accuracy'])

	model.fit(train_data.reshape(-1, width, height, 1), train_labels, epochs=epochs, validation_split=0.2)

	model.save("models/model.h5")
	print("\nModel saved: models/model.h5")

	test_loss, test_acc = model.evaluate(test_data.reshape(-1, width, height, 1), test_labels, verbose=2)
	print("\nTest accuracy:", test_acc)

if __name__ == "__main__":
	argument_parser = argparse.ArgumentParser(description="desc")
	argument_parser.add_argument("--dataset_dir", required=True, help="Dataset folder")
	argument_parser.add_argument("--width", required=True, type=int, help="Image width")
	argument_parser.add_argument("--height", required=True, type=int, help="Image height")
	argument_parser.add_argument("--epochs", required=True, type=int, help="Epochs")
	arguments = argument_parser.parse_args()

	if (True):
		train(
			dataset_dir=arguments.dataset_dir,
			width=arguments.width,
			height=arguments.height,
			epochs=arguments.epochs
		)
