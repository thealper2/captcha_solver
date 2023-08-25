import os
import argparse
from PIL import Image

def preprocess(source_dir, output_dir, width, height):
	file_list = os.listdir(source_dir)
	char_counter = {}

	counter = len(file_list)
	a = 0

	for filename in file_list:
	    if filename.endswith(".png"):
	        image_path = os.path.join(source_dir, filename)
	        image = Image.open(image_path)
	        image_name = os.path.splitext(filename)[0]

	        for i, char in enumerate(image_name):
	            if char not in char_counter:
	                char_counter[char] = 1
	            else:
	                char_counter[char] += 1

	            char_folder = os.path.join(output_dir, char)

	            if not os.path.exists(char_folder):
	                os.makedirs(char_folder)

	            char_image = image.crop((i * width, 0, (i + 1) * width, height))
	            char_image.save(os.path.join(char_folder, f"{char_counter[char]}.png"))
	        a += 1
	        print(f"[-] File: {a}/{counter} processed.")


if __name__ == "__main__":
	argument_parser = argparse.ArgumentParser(description="desc")
	argument_parser.add_argument("--source_dir", required=True, help="Source folder")
	argument_parser.add_argument("--output_dir", required=True, help="Output folder")
	argument_parser.add_argument("--width", required=True, type=int, help="Image width")
	argument_parser.add_argument("--height", required=True, type=int, help="Image height")
	arguments = argument_parser.parse_args()

	if (True):
		preprocess(
			source_dir=arguments.source_dir,
			output_dir=arguments.output_dir,
			width=arguments.width,
			height=arguments.height
		)
