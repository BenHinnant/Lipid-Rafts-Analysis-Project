import skimage.io as skio
import skimage.draw
import skimage.filters
import skimage.measure
import skimage.viewer
import skimage.feature
from skimage import color
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cbook as cbook
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from skimage.util import img_as_int
import imageio
import glob
from PIL import Image, ImageSequence, ImageFilter
from PIL.TiffTags import TAGS
#from PyQt5 import QtCore, QtGui, QtWidgets
import sys
#import os
def main():
    #filename = os.path.join(skimage.data_dir,r'C:\Users\sauro\OneDrive\Documents\Research\Images\w303_pho8gfp_0.4%_2021_08_19__17_39_45_Airyscan Processing-1.tif')
    #filePath = r"C:\Users\sauro\OneDrive\Documents\Research\Images\w303_pho8gfp_0.4%_2021_08_19__17_39_45_Airyscan Processing-1.tif"
    #test_path = r"C:\Users\sauro\OneDrive\Documents\Research\Images\w303_pho8gfp_2021_08_19__17_29_28_Airyscan Processing-1.tif"
    #filename = "Block_21.tif"
    #filename = "Demo_Block_2.tif"
    #filename = "w303_pho8gfp_2021_08_19__17_29_28_Airyscan Processing-1.tif"
    #filename = "w303_pho8gfp_2021_08_19__17_29_28_Airyscan Processing-3.tif"
    #img = mpimg.imread(filePath)
    #parse_tif(filename)
    #save_frame(filename)
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
    #domain_diameters = measure_domain_diameters(image, sigma)
    #create_microdomain_dimensions_file(filename)
    while True:
        function_selected = False
        function_call = input("Please enter a number to select a function: 1 for microdomain dimensions analysis, 2 to save a tif frame, 0 to end the script: ")
        function_num = 0
        try:
            function_num = int(function_call)
        except ValueError:
            print("This is not a number.")
            continue
        if (function_num == 0):
            break
        if (function_num == 1):
            function_selected = True
            image_name = input("What image would you like to analyze? Paste filename or filepath: ")
            image = skio.imread(image_name, as_gray=True)
            sigma = float(input("Enter a sigma value. Decimals are allowed: "))
            scale_bar_question = input("Include a scale bar in your images? y/n: ")
            if (scale_bar_question == "y"):
                include_scale_bar = True
            else:
                include_scale_bar = False
            units_question = input("Which units should the images have? Press 1 for pixels and 2 for microns: ")
            if (int(units_question) == 1):
                unit_type = "pixels"
            elif (int(units_question) == 2):
                unit_type = "microns"
            measure_domain_diameters(image, sigma, image_name, include_scale_bar, unit_type)
        if (function_num == 2):
            function_selected = True
            file_path = input("What image would you like to analyze? Paste filepath or filename: ")
            frame_num = input("Which frame number would you like to save: ")
            frame_name = input("What name do you want for your frame: ")
            save_frame(file_path, frame_num, frame_name)
        if (function_selected == False):
            print ("You have not selected a valid function.")
        
    #issues: the brightest image is not the clearest one because other cells contribute to increasing the brightness. Perhaps restrict the number of objects that can be on screen
    #when determining brightest image, or restrict the location to the center of image, or doing segmentation and choosing the biggest object, or
    #for ideal microdomains, may need to use erosion to make finer lines  
    #future issues: only use XResolution and not YResolution, currently removes the largest region assuming its the background
'''    
def create_microdomain_dimensions_file(filename, sigma):
    brightest_img = find_brightest_frame(filename, sigma)
    obj_major_axis_length, obj_minor_axis_length = measure_domain_diameters(brightest_img)
    print("Major axis lengths: ", obj_major_axis_length)
    print("Minor axis lengths: ", obj_minor_axis_length)
'''
def find_brightest_frame(filename, sigma):
    #image = Image.open(filename)
    #greyscale_img = image.convert("L")
    #img = greyscale_img.filter(ImageFilter.GaussianBlur(radius = 3.5))
    #img = img.filter(ImageFilter.FIND_EDGES)
    #img.show()
    brightest_img = []
    brightest_img_brightness = 0
    brightest_img_index = 0
    brightness_arr = []

    with Image.open(filename) as img:
        for i, frame in enumerate(ImageSequence.Iterator(img)):
            current_frame = frame.copy()
            current_frame = current_frame.convert("L")
            if i == 3:
                canny_viewer(np.array(current_frame))
            current_frame = current_frame.filter(ImageFilter.GaussianBlur(radius = 3.5))
            current_frame = current_frame.filter(ImageFilter.FIND_EDGES)
            #current_frame.show()
            frame_brightness = measure_frame_brightness(np.array(current_frame), sigma)
            brightness_arr.append(frame_brightness)
            if frame_brightness > brightest_img_brightness:
                brightest_img_index = i
                brightest_img_brightness = frame_brightness
                brightest_img = np.array(current_frame)
    print (brightness_arr)
    print ("Brightest image index: ", brightest_img_index)
    plt.imshow(brightest_img)
    plt.show()
    return brightest_img
    
def canny_edge_detection(filename, sigma, low_threshold, high_threshold):
    image = skimage.io.imread(fname=filename, as_gray=True)
    skimage.io.imshow(image)
    edges = skimage.feature.canny(
        image=image,
        sigma=sigma,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )
    skio.imshow(edges)

def canny_viewer(image):
    #image = skimage.io.imread(fname=filename, as_gray=True)
    #viewer = skimage.io.imshow(image)
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
    gray_img = color.rgb2gray(image)
    viewer = skimage.viewer.ImageViewer(image=gray_img)
    viewer += canny_plugin
    viewer.show()

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
    for i in range (0,3):
        try:
            img.seek(i)
            img.save('Demo_Block_%s.tif'%(i,))
        except EOFError as e:
            print(e)

def save_frame(file_path, frame_num, frame_name):
    img = Image.open(file_path)
    numFramesPerTif = img.n_frames
    for i in range (numFramesPerTif):
        try:
            if i == 3:
                img.save('Hyesoo_Block_%s.tif'%(i,))    
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

# Measures the microdomain domains X and Y axes in microns 
def measure_domain_diameters(image, sigma=3.5, image_filename="null", include_scale_bar=False, unit_type="pixels", connectivity=2):
    blurred_image = skimage.filters.gaussian(image/image.max(), sigma=sigma)
    threshold = skimage.filters.threshold_otsu(blurred_image)
    print("Threshold: ", threshold)
    binary_mask = blurred_image < threshold
    labeled_image, count = skimage.measure.label(binary_mask, connectivity=connectivity, return_num=True)
    print("# of microdomains: ", count-1) 
    obj_features = skimage.measure.regionprops(labeled_image)
    del obj_features[0]
    obj_major_axis_pixels = [objf["axis_major_length"] for objf in obj_features]
    obj_minor_axis_pixels = [objf["axis_minor_length"] for objf in obj_features]
    if (unit_type == "pixels"):
        print("Major axis lengths in pixels: ", obj_major_axis_pixels)
        print("Minor axis lengths in pixels: ", obj_minor_axis_pixels)
    if (unit_type == "microns"):
        # Given the image filename, finds the X resolution to convert pixels to microns
        physical_size_x, physical_size_x_unit = find_resolution(image_filename)
        obj_major_axis_microns = [float(x)*physical_size_x for x in obj_major_axis_pixels]
        obj_minor_axis_microns = [float(x)*physical_size_x for x in obj_minor_axis_pixels]
        print("Major axis lengths in microns: ", obj_major_axis_microns)
        print("Minor axis lengths in microns: ", obj_minor_axis_microns)
    #obj_diameter = [objf["equivalent_diameter_area"] for objf in obj_features]
    #print("Diameters", obj_diameter)
    if (include_scale_bar == True):
        physical_size_x, physical_size_x_unit = find_resolution(image_filename)
        show_scale_bar_image(image, physical_size_x, physical_size_x_unit)
        show_scale_bar_image(blurred_image, physical_size_x, physical_size_x_unit)
        show_scale_bar_image(binary_mask, physical_size_x, physical_size_x_unit)
    else:
        plt.imshow(image)
        plt.show()
        plt.imshow(blurred_image)
        plt.show()
        plt.imshow(binary_mask)
        plt.show()
    #return obj_major_axis_microns, obj_minor_axis_microns
    
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

#Given a filename, finds the x resolution of the image, e.g. 3 microns per pixel.
#Assumes the y resolution is the same

def find_resolution(image_filename):
    with Image.open(image_filename) as img:
        meta_dict = {TAGS[key] : img.tag[key] for key in img.tag.keys()}
        #print(meta_dict)
        #print(type(meta_dict))
        #print(meta_dict["XResolution"])
        #print(type(meta_dict["XResolution"]))
        xresolution = meta_dict["XResolution"][0]
        physical_size_x = xresolution[0]/xresolution[1]
        #print(physical_size_x)
        #print(type(physical_size_x))
        physical_size_x_unit = "um"
        return physical_size_x, physical_size_x_unit

'''
def find_resolution(image_filename):
    print (image_filename)
    with tifffile.TiffFile(image_filename) as tif:
            imagej_metadata = tif.imagej_metadata
    print (imagej_metadata)
    xres_meters = imagej_metadata['XResolution']
    xres_microns = xres_meters*(10^6)
    #physical_size_y = metadata['PhysicalSizeY']
    xres_unit = imagej_metadata['Unit']
    print (xres_microns.type)
    print (xres_unit.type)
    return xres_microns, xres_unit
'''
# Given an image, show that image with a scalebar
def show_scale_bar_image(image, physical_size_x, physical_size_x_unit):
    fontprops = fm.FontProperties(size=10)
    fig, ax = plt.subplots()
    ax.imshow(image)
    scalebar = AnchoredSizeBar(ax.transData,
                           10, '10 ' + physical_size_x_unit, 'lower right', 
                           pad=0.1,
                           color='white',
                           frameon=False,
                           size_vertical=1,
                           fontproperties=fontprops)
    ax.add_artist(scalebar)
    plt.show()

if __name__ == '__main__':
    main()