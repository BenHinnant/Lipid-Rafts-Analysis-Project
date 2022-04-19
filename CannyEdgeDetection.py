import skimage.io
import skimage.feature
import sys

filename = sys.argv[1]
sigma = float(sys.argv[2])
low_threshold = float(sys.argv[3])
high_threshold = float(sys.argv[4])
image = skimage.io.imread(fname=filename, as_gray=True)
skimage.io.imshow(image)
edges = skimage.feature.canny(
    image=image,
    sigma=sigma,
    low_threshold=low_threshold,
    high_threshold=high_threshold,
)
skimage.io.imshow(edges)