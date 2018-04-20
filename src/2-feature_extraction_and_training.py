#!/usr/bin/python

from skimage import io
from skimage import filters
# Adapt Grey Scale Filters to RGB
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
from skimage import restoration
from sklearn import linear_model
from skimage import feature
from skimage import morphology
from skimage import img_as_float
from skimage import img_as_uint
from sklearn.linear_model import LogisticRegression
import scipy
import pywt


classes = ['Nex7','HTC-1-M7','iP4s','iP6','LG5x','MotoMax','MotoNex6','MotoX','GalaxyN3','GalaxyS4']

global_disk_radius = morphology.disk(3)

@adapt_rgb(each_channel)
def median_each(image):
    return filters.median(image)

@adapt_rgb(each_channel)
def mean_each(image):
    return filters.rank.mean(image,global_disk_radius)
    
@adapt_rgb(each_channel)
def entropy_each(image):
    return filters.rank.entropy(image,global_disk_radius)

# @adapt_rgb(each_channel)
# def lbp_each(image):
#     return feature.local_binary_pattern(image)

super_mean_rank_accumulator = {}
super_entropy_rank_accumulator = {}
number_of_samples = {}

def process_based_on_noise(noised_image):#,device):
    # Access global variables
    global super_mean_rank_accumulator
    global number_of_samples

    # May be useful to change disk size (maybe?)
    # ValueError: Images of type float must be between -1 and 1.
    #noised_image = img_as_float(noised_image)
    m_mean_rank = mean_each(noised_image)
    m_entropy_rank = entropy_each(noised_image)

    # ============================================================
    # ========= Finding Core Features - Compare Distance KNN? ====
    # ============================================================
    # May be useful to change the importance (feature weight) of these guys below
    # number_of_samples[device] += 1
    # if not device in super_mean_rank_accumulator:
    #     super_mean_rank_accumulator[device] = m_mean_rank
    #     super_entropy_rank_accumulator[device] = m_entropy_rank
    # else
    #     super_mean_rank_accumulator[device] += m_mean_rank
    #     super_entropy_rank_accumulator[device] += m_entropy_rank

    # Return a list with all found features 

    return m_mean_rank.tolist()+ m_entropy_rank.tolist()

def extract_features_from_image(image):
    feature_list = []
    # Get median version of the image
    median_version = image - median_each(image)
    for i in range(0,3):
        LL, HIGH_VALUES = pywt.dwt2(median_version[...,i],'haar')
        for item in HIGH_VALUES:
            data_description = scipy.stats.describe(item)
            # Mean
            feature_list.append(min(data_description[2]))
            feature_list.append(max(data_description[2]))
            feature_list.append(sum(data_description[2])/float(data_description[0]))
            # Variance
            feature_list.append(min(data_description[3]))
            feature_list.append(max(data_description[3]))
            feature_list.append(sum(data_description[3])/float(data_description[0]))
            # Skewness
            feature_list.append(min(data_description[4]))
            feature_list.append(max(data_description[4]))
            feature_list.append(sum(data_description[4])/float(data_description[0]))
            # Kurtosis
            feature_list.append(min(data_description[5]))
            feature_list.append(max(data_description[5]))
            feature_list.append(sum(data_description[5])/float(data_description[0]))
        
    #feature_list = process_based_on_noise(median_version)
    # Get bilateral version of the image - NOT WORKING WELL
    # bilateral_version = image - restoration.denoise_bilateral(image,bins=3)
    # feature_list.extend(process_based_on_noise(bilateral_version))
    # Get wavelet_version of the image -- IS THIS REALLY NECESSARY?
    # wavelet_version = image - restoration.denoise_wavelet(image)
    # feature_list.extend(process_based_on_noise(wavelet_version))
    return feature_list
    
def main():
    # Access global variables
    global super_mean_rank_accumulator
    global number_of_samples
    
    lr_model = LogisticRegression()

    features = []
    devices = []

    for device in classes:
        for i in range(1,11):
            # Load image as ndarray - N dimensions array
            file_name = '../input512/('+device+')'+str(i)+'.tiff'
            image = io.imread(file_name)

            # Increase the number of samples per device would be an option!!

            # =======================================================
            # ========= Generating the three types of image =========
            # =======================================================
            print('Processing image '+file_name)
            feature_list = extract_features_from_image(image)
            features.append(feature_list)
            devices.append(device)
            
            # Get LBP version of the image - NOT INTERESTING, SAY THIS: are the binary representation of the texture fluctuation around each point, LOOK WHAT IS LBP 4 REAL
            # lbp_version = image - lbp_each(image)

            # Training the LogisticRegression
    lr_model.fit(features,devices)

    # Single test
    test_images = []
    test_image = io.imread('../input512/(iP6)1.tiff')
    test_images.append(extract_features_from_image(test_image))
    test_image = io.imread('../input512/(MotoX)3.tiff')
    test_images.append(extract_features_from_image(test_image))
    test_image = io.imread('../input512/(Nex7)8.tiff')
    test_images.append(extract_features_from_image(test_image))
    print(lr_model.predict(test_images))
        
if __name__ == '__main__':
    main()        
        

