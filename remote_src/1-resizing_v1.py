#!/usr/bin/python

from skimage import io
from skimage import filters
# Adapt Grey Scale Filters to RGB
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
from skimage import restoration
from sklearn import linear_model

classes = ['Nex7','HTC-1-M7','iP4s','iP6','LG5x','MotoMax','MotoNex6','MotoX','GalaxyN3','GalaxyS4']

# Must use to rename Next7
# rename 's/JPG/jpg/g' ../input/Nex7/*


for device in classes:
    for i in range(1,276):
        # Load image as ndarray - N dimensions array
        
        file_name = '../original_files/unpacked_train/('+device+')'+str(i)+'.jpg'
        image = io.imread(file_name)
        if len(image.shape) == 1:
            image = image[0]
        # Crop image in certain params to find the mid 512x512 aspect
        x1 = max(image.shape[0]/2-256,0)
        x2 = min(x1+512,image.shape[0])
        y1 = max(image.shape[1]/2-256,0)
        y2 = min(y1+512,image.shape[1])
        image = image[x1:x2,y1:y2]
        input_512_file_name = '../training/('+device+')'+str(i)+'.tif'
        print('Saving image '+input_512_file_name)
        io.imsave(input_512_file_name,image)
        # median_version_r = image - median_each(image)
        # bilateral_version = image - restoration.denoise_bilateral(image,multichannel=True)
        # wavelet_version = image - restoration.denoise_wavelet(image)






