import cv2
import numpy as np
from skimage import io, color, measure

#---------------
dst_path = 'black_white.jpg' # frame of the video
image = io.imread(dst_path)

# Label connected components
gray_image = color.rgb2gray(image)
image_mask = gray_image > .5
labeled_image = measure.label(image_mask,connectivity=1)

# Find and display connected components

lines = []
regions = measure.regionprops(labeled_image)
for region in regions:
    val = labeled_image[region.coords[0,0],region.coords[0,1]]

    filtered_mask= np.where(labeled_image == val)

    bbox_ratio_comp = 1-region.area_filled/region.area_bbox #Take the compliment of the ratio of pixel_area/bbox_area 

    if( bbox_ratio_comp > .8): # High bbox_ratio_comp suggests a lot of empty space, meaning only thin lines should appear here.
        lines.append(val)
    else:
        filtered_mask= np.where(labeled_image == val)
        labeled_image[filtered_mask[0],filtered_mask[1]] = 0 #set those outliers as background 
        
print(lines)
io.imshow(image)
io.imshow(labeled_image, cmap='nipy_spectral', alpha=.5)  # Adjust cmap as needed
io.show()