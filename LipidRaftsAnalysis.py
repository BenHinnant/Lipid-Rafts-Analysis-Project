import skimage.io as skio
import skimage.draw
import skimage.filters
import skimage.measure
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.util import img_as_int
import imageio
import glob
from PIL import Image, ImageSequence
sigma = 3.5
min_area = 1000
#import sys
#import os
def main():
    #filename = os.path.join(skimage.data_dir,r'C:\Users\sauro\OneDrive\Documents\Research\Images\w303_pho8gfp_0.4%_2021_08_19__17_39_45_Airyscan Processing-1.tif')
    filePath = r"C:\Users\sauro\OneDrive\Documents\Research\Images\w303_pho8gfp_0.4%_2021_08_19__17_39_45_Airyscan Processing-1.tif"
    test_path = r"C:\Users\sauro\OneDrive\Documents\Research\Images\w303_pho8gfp_2021_08_19__17_29_28_Airyscan Processing-1.tif"
    #img = mpimg.imread(filePath)
    #parse_tif(filePath)
    #filename = "Block_21.tif"
    filename = "w303_pho8gfp_2021_08_19__17_29_28_Airyscan Processing-1.tif"
    #image = skio.imread(filename, as_gray=True)
    #print (image.shape)
    #skio.imshow(image/image.max())
    #skio.show()
    #blurred_image = gaus_blur(filename, sigma)
    #plt.imshow(blurred_image)
    #plt.show()
    #show_grayscale_histogram(blurred_image)
    #thresholding(image/image.max(), 0.25)
    #auto_thresholding(blurred_image/blurred_image.max())
    #all_images = glob.glob("Block_*")
    #for filename in all_images:
    #    density = measure_image_brightness(filename, 3.5)
    #    print(filename, density, sep=",")
    #labeled_image, count = connected_component_analysis(filename, sigma, threshold=0.2, connectivity=2)
    #plt.imshow(labeled_image)
    #plt.show()
    #print("dtype:", labeled_image.dtype)
    #print("min:", np.min(labeled_image))
    #print("max:", np.max(labeled_image))
    #print("# of objects: ", count) 
    #colored_label_image = skimage.color.label2rgb(labeled_image, bg_label=0)
    #plt.imshow(colored_label_image)
    #plt.show()
    #obj_areas = measure_object_areas(labeled_image)
    #area_histogram(obj_areas)
    #measure_cells(labeled_image, min_area)
    #measure_domain_diameters(filename, sigma)
    return_csv_data(test_path)
    
def return_csv_data(filename):
    img = Image.open(filename)
    brightest_img_brightness = 0
    brightest_img_index = 0
    membrane_densities = []
    images = []
    for i in range (img.n_frames):
        img.seek(i)
        images.append()

    #frame = img.n_frames
    #with Image.open(filename) as img:
     #   for i, frame in enumerate(ImageSequence.Iterator(img)):
      #      current_frame = frame.copy()
       #     print (type(frame))
        #    print (type(current_frame))
         #   frame_brightness = measure_frame_brightness(current_frame, sigma)
          #  if frame_brightness > brightest_image_brightness:
           #     brightest_image_index = i
   
    for i in range (img.n_frames):
        try:
            img.seek(i)
            img.save('Frame_%s_' % (i), filename)
            brightest_frame_filename = ('Frame_%s_' % (i), filename)
            img_brightness = measure_image_brightness(brightest_frame_filename, sigma)
            if img_brightness > brightest_img_brightness:
                brightest_img_index = i
            membrane_densities[i] = img_brightness
        except EOFError as e:
           print(e)
    print (membrane_densities)
    print ("Brightest image index: ", brightest_img_index)

def convert_img_to_np_array(path):
    img = Image.open(path)
    images = []
    for i in range (img.n_frames):
        img.seek(i)
        images.append(np.array(img))
    return np.array(images)

def parse_tif(filePath):
    img = Image.open(filePath)
    numFramesPerTif = img.n_frames
    for i in range (numFramesPerTif):
        try:
            img.seek(i)
            img.save('Block_%s.tif'%(i,))
        except EOFError as e:
            print(e)

def gaus_blur(filename, sigma):
    image = skio.imread(fname=filename)
    blur = skimage.filters.gaussian(image, sigma=(sigma,sigma), truncate=3.5)
    adjusted_blur = blur/blur.max()
    skio.imshow(adjusted_blur)
    skio.show()
    return (adjusted_blur)

def show_grayscale_histogram(image):
    histogram, bin_edges = np.histogram(image, bins=256, range=(0.0, 1.0))
    plt.plot(bin_edges[0:-1], histogram)
    plt.title("Grayscale Histogram")
    plt.xlabel("grayscale value")
    plt.ylabel("pixels")
    plt.xlim(0, 1.0)
    plt.show()

def thresholding(image, threshold):
    binary_mask = image > threshold
    plt.imshow(binary_mask, cmap='gray')
    plt.show()

def auto_thresholding(image):
    threshold = skimage.filters.threshold_otsu(image)
    print("Found automatic threshold = {}.".format(threshold)) 
    binary_mask = image > threshold
    plt.imshow(binary_mask, cmap='gray')
    plt.show()
    foreground = np.zeros_like(image)
    foreground[binary_mask] = image[binary_mask]
    plt.imshow(foreground)
    plt.show()

def measure_image_brightness(filename, sigma=3.5):
    image = skio.imread(filename, as_gray=True)
    blurred_image = skimage.filters.gaussian(image, sigma=sigma)
    threshold = skimage.filters.threshold_otsu(blurred_image)
    binary_mask = blurred_image > threshold
    memPixels = np.count_nonzero(binary_mask)
    w = binary_mask.shape[1]
    h = binary_mask.shape[0]
    density = memPixels/(w*h)
    return density

def measure_frame_brightness(image, sigma=3.5):
    blurred_image = skimage.filters.gaussian(image, sigma=sigma)
    threshold = skimage.filters.threshold_otsu(blurred_image)
    binary_mask = blurred_image > threshold
    memPixels = np.count_nonzero(binary_mask)
    w = binary_mask.shape[1]
    h = binary_mask.shape[0]
    density = memPixels/(w*h)
    return density

def connected_component_analysis(filename, sigma=3.5, threshold=0.2, connectivity=2):
    image = skio.imread(filename, as_gray=True)
    blurred_image = skimage.filters.gaussian(image/image.max(), sigma=sigma)
    threshold = skimage.filters.threshold_otsu(blurred_image)
    print("Threshold: ", threshold)
    binary_mask = blurred_image > threshold
    plt.imshow(binary_mask)
    plt.show()
    labeled_image, count = skimage.measure.label(binary_mask, connectivity=connectivity, return_num=True)
    return labeled_image, count

def measure_domain_diameters(filename, sigma=3.5, connectivity=2):
    image = skio.imread(filename, as_gray=True)
    blurred_image = skimage.filters.gaussian(image/image.max(), sigma=sigma)
    threshold = skimage.filters.threshold_otsu(blurred_image)
    print("Threshold: ", threshold)
    binary_mask = blurred_image < threshold
    plt.imshow(binary_mask)
    plt.show()
    labeled_image, count = skimage.measure.label(binary_mask, connectivity=connectivity, return_num=True)
    print("# of objects: ", count) 
    obj_features = skimage.measure.regionprops(labeled_image)
    obj_major_axis_length = [objf["axis_major_length"] for objf in obj_features]
    obj_minor_axis_length = [objf["axis_minor_length"] for objf in obj_features]
    obj_diameter = [objf["equivalent_diameter_area"] for objf in obj_features]
    print("Major axis lengths: ", obj_major_axis_length)
    print("Minor axis lengths: ", obj_minor_axis_length)
    print("Diameters", obj_diameter)
    
def measure_object_areas(labeled_image):
    obj_features = skimage.measure.regionprops(labeled_image)
    obj_areas = [objf["area"] for objf in obj_features]
    print(obj_areas)
    return obj_areas

def area_histogram(obj_areas):
    plt.hist(obj_areas)
    plt.xlabel("Area (pixels)")
    plt.ylabel("# of objects")
    plt.show()

def measure_cells(labeled_image, min_area):
    obj_features = skimage.measure.regionprops(labeled_image)
    large_objects = []
    for objf in obj_features:
        if objf["area"] > min_area:
            large_objects.append(objf["area"])
    print("Found", len(large_objects), "cell(s)")

if __name__ == '__main__':
    main()