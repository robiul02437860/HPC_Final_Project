import matplotlib.pyplot as plt
import numpy as np

def read_pgm(filename):
    with open(filename, 'rb') as f:
        # Read header
        magic = f.readline()
        if magic != b'P5\n':
            raise ValueError("Not a valid binary PGM file.")

        # Read width, height
        while True:
            line = f.readline()
            if line.startswith(b'#'):
                continue
            else:
                width, height = [int(i) for i in line.strip().split()]
                break

        # Read max gray value
        maxval = int(f.readline().strip())

        # Read pixel data
        img = np.frombuffer(f.read(), dtype=np.uint8).reshape((height, width))

    return img

def visualize(original_path, segmented_path=None):
    img1 = read_pgm(original_path)

    if segmented_path:
        img2 = read_pgm(segmented_path)

        # Show side-by-side
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img1, cmap='gray')
        plt.title("Original")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(img2, cmap='gray')
        plt.title("Segmented")
        plt.axis('off')

    else:
        plt.imshow(img1, cmap='gray')
        plt.title("PGM Image")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    visualize("data/input.pgm", "results/output.pgm")
