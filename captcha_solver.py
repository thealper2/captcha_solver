import cv2
import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


def predict(file, width, height, model_path):
	model = load_model(model_path)

	characters = os.listdir("extract_files")

	captcha_image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
	captcha_data = captcha_image.reshape(-1, width * 4, height, 1) / 255.0
	captcha_result = ""
	for i in range(4):
	    char_data = captcha_data[:, i * width:(i + 1) * width, :]
	    prediction = model.predict(char_data)
	    predicted_label = np.argmax(prediction, axis=1)
	    predicted_char = characters[predicted_label[0]]
	    captcha_result += predicted_char
	print("CAPTCHA: ", captcha_result)


if __name__ == "__main__":
	argument_parser = argparse.ArgumentParser(description="desc")
	argument_parser.add_argument("--file", required=True, help="File")
	argument_parser.add_argument("--width", required=True, type=int, help="Image width")
	argument_parser.add_argument("--height", required=True, type=int, help="Image height")
	argument_parser.add_argument("--model_path", required=True, help="Model path")
	arguments = argument_parser.parse_args()

	if True:
		predict(
			file=arguments.file,
			width=arguments.width,
			height=arguments.height,
			model_path=arguments.model_path
		)
