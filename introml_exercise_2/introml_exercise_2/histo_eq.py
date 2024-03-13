# Implement the histogram equalization in this file
import numpy as np
import matplotlib.pyplot as plt
import cv2

#Load hello.png into a numpy array, using, for example, PIL or opencv
img = cv2.imread('introml_exercise_2\introml_exercise_2\hello.png', cv2.IMREAD_GRAYSCALE)
#print(img)

#Compute the intensity histogram of your cat image, for pixel values between 0 and 255
img_flattened_array = img.flatten()  #We can use flatten method to get a copy of array collapsed into one dimension  
#print(img_flattened_array)
histogram = np.zeros(256)
#print(histogram)
for i in img_flattened_array:
        histogram[i] += 1
print(np.sum(histogram[:90]))
#plt.plot(histogram)
#plt.show()

#Compute its cumulative distribution function C
cum_freq = np.zeros(256)
cum_freq[0] = histogram[0]
for i in range(1, len(histogram)-1):
    cum_freq[i] = cum_freq[i-1] + histogram[i]

cdf = cum_freq/max(cum_freq)
#print(np.sum(cdf[:90]))
'''
dp  c    pdf   cf    cdf
1   5    .10    5   .10 = cf/max(cf i.e 50) 
2   15   .30   20   .40
3   10   .20   30   .60
11  20   .40   50   1.00
'''
#plt.plot(cdf)
#plt.show()

#Change the gray value of each pixel
size = img_flattened_array.size
new_npArry_CGV = np.zeros(size)
#print(new_npArry_CGV)

for i in range(len(img_flattened_array)):
    pixel_value = img_flattened_array[i]
    new_npArry_CGV[i] = (((cdf[pixel_value])-cdf.min())/ (1- cdf.min()))*255
    #print(new_npArry_CGV)
CGV_img = new_npArry_CGV.reshape(img.shape)
plt.imshow(CGV_img, 'gray')
plt.show()

#Save the result as kitty.png
cv2.imwrite("kitty.png", CGV_img)

#During the submission, explain why the background looks like this, i.e., with clear intensity boundaries
#redistribute of intensities with histogram equalization
#from cdf we do rebin-< spreadout intensities
#cdf from pdff and map the values
'''
Approximately uniform distribution of the intensity values
• Pixels are spread evenly across the entire range of intensity values
• Highest possible contrast
Often produces unrealistic looking images or undesirable effects
• When image contains regions that are significantly lighter/darker than most of the image
: contrast in those regions will not be sufficiently enhanced
'''
