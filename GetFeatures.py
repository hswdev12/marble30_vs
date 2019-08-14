from PIL import Image
import numpy as np
from skimage.feature import local_binary_pattern, greycomatrix, greycoprops
from skimage.filters import gabor
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import pandas as pd
import sys

file_name = sys.argv[1]  # './similars/image82.png'    #
# print(file_name)
# Feature extraction on single image
img = Image.open(file_name)  # PIL image
img_gray = img.convert('L')  # Converting to grayscale
img_arr = np.array(img_gray)  # Converting to array

# Finding LBP
# Radius = 1, No. of neighbours = 8
feat_lbp = local_binary_pattern(img_arr, 8, 1, 'uniform')
feat_lbp = np.uint8((feat_lbp/feat_lbp.max())*255)  # Converting to unit8
lbp_img = Image.fromarray(feat_lbp)  # Conversion from array to PIL image

# Energy and Entropy of LBP feature
lbp_hist, _ = np.histogram(feat_lbp, 8)
lbp_hist = np.array(lbp_hist, dtype=float)
lbp_prob = np.divide(lbp_hist, np.sum(lbp_hist))
lbp_energy = np.sum(lbp_prob**2)
lbp_entropy = -np.sum(np.multiply(lbp_prob, np.log2(lbp_prob)))
# print('LBP energy = '+str(lbp_energy))
# print('LBP entropy = '+str(lbp_entropy))

# Finding GLCM features from co-occurance matrix
# Co-occurance matrix
gCoMat = greycomatrix(img_arr, [2], [0], 256, symmetric=True, normed=True)
contrast = greycoprops(gCoMat, prop='contrast')
dissimilarity = greycoprops(gCoMat, prop='dissimilarity')
homogeneity = greycoprops(gCoMat, prop='homogeneity')
energy = greycoprops(gCoMat, prop='energy')
correlation = greycoprops(gCoMat, prop='correlation')
# print('Contrast = '+str(contrast[0][0]))
# print('Dissimilarity = '+str(dissimilarity[0][0]))
# print('Homogeneity = '+str(homogeneity[0][0]))
# print('Energy = '+str(energy[0][0]))
# print('Correlation = '+str(correlation[0][0]))

feat_glcm = np.array([contrast[0][0], dissimilarity[0][0],
                      homogeneity[0][0], energy[0][0], correlation[0][0]])

# Gabor filter
gaborFilt_real, gaborFilt_imag = gabor(img_arr, frequency=0.6)
gaborFilt = (gaborFilt_real**2+gaborFilt_imag**2)//2
# Displaying the filter response

# Energy and Entropy of Gabor filter response
gabor_hist, _ = np.histogram(gaborFilt, 8)
gabor_hist = np.array(gabor_hist, dtype=float)
gabor_prob = np.divide(gabor_hist, np.sum(gabor_hist))
gabor_energy = np.sum(gabor_prob**2)
gabor_entropy = -np.sum(np.multiply(gabor_prob, np.log2(gabor_prob)))
# print('Gabor energy = '+str(gabor_energy))
# print('Gabor entropy = '+str(gabor_entropy))
# Concatenating features(2+5+2)
concat_feat = np.concatenate(([lbp_energy, lbp_entropy], feat_glcm, [
                             gabor_energy, gabor_entropy]), axis=0)
# print(concat_feat)

# returnValue = '*'.join(str(concat_feat))
# for item in concat_feat:
#     returnValue = '*' + returnValue + str(item)
return_list = [file_name]
for item in concat_feat:
    return_list.append(str(item))

return_value = '*'.join(return_list)
print(return_value)
