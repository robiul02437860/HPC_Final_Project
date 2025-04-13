import sys
import numpy as np

def read_pgm(filename):
    with open(filename, 'rb') as f:
        assert f.readline() == b'P5\n'
        dims = f.readline()
        while dims.startswith(b'#'):
            dims = f.readline()
        width, height = map(int, dims.split())
        maxval = int(f.readline())
        img = np.fromfile(f, dtype=np.uint8, count=width * height)
        return img.reshape((height, width))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 validation.py output1.pgm output2.pgm")
        sys.exit(1)

    img1 = read_pgm(sys.argv[1])
    img2 = read_pgm(sys.argv[2])

    if np.array_equal(img1, img2):
        print("Validation Passed!")
    else:
        print("Validation Failed!")
