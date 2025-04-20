from PIL import Image
import numpy as np

def convert_color_to_pgm(input_path, output_path, resize_width, resize_height):
    try:
        # Load, resize, and convert to grayscale
        img = Image.open(input_path)
        img_resized = img.resize((resize_width, resize_height), Image.ANTIALIAS)
        img_gray = img_resized.convert('L')
        gray_array = np.array(img_gray)

        # Write as binary PGM (P5)
        with open(output_path, 'wb') as f:
            f.write(f'P5\n{resize_width} {resize_height}\n255\n'.encode())
            f.write(gray_array.tobytes())

        print(f"Saved PGM: {output_path}")

    except Exception as e:
        print(f"Error: {e}")

# Example usage
if __name__ == "__main__":
    convert_color_to_pgm("data/original/input.jpg", "data/input.pgm", 1024, 1024)
