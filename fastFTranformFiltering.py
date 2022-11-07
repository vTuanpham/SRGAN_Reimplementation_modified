
import cv2
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


img = cv.imread('unDenoiseSuperres.jpg',0)
f = cv2.dft(np.float32(img),flags=cv2.DFT_COMPLEX_OUTPUT)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(cv2.magnitude(fshift[:,:,0],fshift[:,:,1])+1)

rows, cols = img.shape
crow, ccol = int(rows/2), int(cols/2)

mask = np.ones((rows, cols, 2), np.uint8)

#Circle
r = int(np.max(img.shape) * .28)
center = [crow, ccol]
x, y = np.ogrid[:rows, :cols]
mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 >= r*r
mask[mask_area] = 0


#Gradient circle
x_axis = np.linspace(-1, 1, img.shape[1])
y_axis = np.linspace(-1, 1, img.shape[0])

xx, yy = np.meshgrid(x_axis, y_axis)
arr = np.sqrt(xx ** 2 + yy ** 2)-5
arr = np.expand_dims(arr,axis=2)


plt.imshow(arr, cmap='gray')
plt.show()
# mask[arr] = 0


fshift_masked = fshift * mask

# fshift_masked = fshift / arr * mask
magnitude_spectrum_masked = 20*np.log(cv2.magnitude(fshift_masked[:,:,0],fshift_masked[:,:,1])+1)
fishift_masked = np.fft.ifftshift(fshift_masked)
img_back = cv2.idft(fishift_masked)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

# cv2.imshow('test',np.array(img_back,dtype=np.uint8))
# cv2.imwrite('fFtTranformMaskedImg.jpg',np.array(img_back))

plt.subplot(141),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.subplot(142),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

plt.subplot(143),plt.imshow(magnitude_spectrum_masked, cmap = 'gray')
plt.title('Magnitude Spectrum masked'), plt.xticks([]), plt.yticks([])

plt.subplot(144),plt.imshow(img_back, cmap = 'gray')
plt.title('img_back'), plt.xticks([]), plt.yticks([])

plt.show()

# def dft_converter(img):
#     img = cv.imread('unDenoiseSuperres.jpg')
#     img_b = img[:,:,0]
#     img_g = img[:,:,1]
#     img_r = img[:,:,2]
#
#     def convertdft(img):
#         f = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
#         fshift = np.fft.fftshift(f)
#         magnitude_spectrum = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]) + 1)
#
#         rows, cols = img.shape
#         crow, ccol = int(rows / 2), int(cols / 2)
#
#         mask = np.ones((rows, cols, 2), np.uint8)
#         r = 500
#         center = [crow, ccol]
#         x, y = np.ogrid[:rows, :cols]
#         mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 >= r * r
#         mask[mask_area] = 0
#
#         fshift_masked = fshift * mask
#         magnitude_spectrum_masked = 20 * np.log(cv2.magnitude(fshift_masked[:, :, 0], fshift_masked[:, :, 1]) + 1)
#         fishift_masked = np.fft.ifftshift(fshift_masked)
#         img_back = cv2.idft(fishift_masked)
#         img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
#
#         img_back = cv2.normalize(img_back,None,0,1,cv2.NORM_MINMAX)
#         # cv.imshow('test',img_back)
#
#         # plt.subplot(141), plt.imshow(img, cmap='gray')
#         # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
#         #
#         # plt.subplot(142), plt.imshow(magnitude_spectrum, cmap='gray')
#         # plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
#         #
#         # plt.subplot(143), plt.imshow(magnitude_spectrum_masked, cmap='gray')
#         # plt.title('Magnitude Spectrum masked'), plt.xticks([]), plt.yticks([])
#         #
#         # plt.subplot(144), plt.imshow(img_back, cmap='gray')
#         # plt.title('img_back'), plt.xticks([]), plt.yticks([])
#         #
#         # plt.show()
#
#         return img_back
#     # cv2.imshow('test',np.array(img_back,dtype=np.uint8))
#     # cv2.imwrite('fFtTranformMaskedImg.jpg',np.array(img_back))
#
#
#     deNoise_imgb = convertdft(img_b)
#     deNoise_imgg = convertdft(img_g)
#     deNoise_imgr = convertdft(img_r)
#
#     deNoise_img = cv.merge((deNoise_imgb,deNoise_imgg,deNoise_imgr))
#     cv.imshow('final',deNoise_img)
#     cv.waitKey(10000)
#
# dft_converter(1)