import skimage.io
import skimage.feature
import skimage.viewer
import sys


filename = sys.argv[1]
image = skimage.io.imread(fname=filename, as_gray=True)
viewer = skimage.io.imshow(image)

canny_plugin = skimage.viewer.plugins.Plugin(image_filter=skimage.feature.canny)
canny_plugin.name = "Canny Filter Plugin"

canny_plugin += skimage.viewer.widgets.Slider(
    name="sigma", low=0.0, high=7.0, value=2.0
)
canny_plugin += skimage.viewer.widgets.Slider(
    name="low_threshold", low=0.0, high=1.0, value=0.1
)
canny_plugin += skimage.viewer.widgets.Slider(
    name="high_threshold", low=0.0, high=1.0, value=0.2
)

viewer += canny_plugin
viewer.show()