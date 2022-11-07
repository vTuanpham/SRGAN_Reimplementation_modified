import math

from keras.models import load_model
from numpy.random import randint
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import tensorflow as tf
from CustomPadding import SymmetricPadding2D
from PIL import ImageEnhance, Image
from skimage import exposure

#SymmetricPadding2D need to be specified as custom in order to load
# generator = load_model('gen_e_12 23.80226371314068 25000 [0.59059704 2.49955556].h5', compile=False,custom_objects={"SymmetricPadding2D": SymmetricPadding2D})

# def adjust_shadow(img):
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
#     v = hsv[:,:,2]
#     print(v)
#
#     # expV = np.vectorize(math.exp)
#
#     # v = np.add(np.log2(np.add(255,-v),v)-1)
#
#     v = np.add(np.square(np.add(255,-v)) * 0.005,v)
#
#     print(v)
#     hsv[:,:,2] = v
#     img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
#
#     return img
#
# def adjust_brightness(img):
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
#     v = hsv[:,:,2]
#     print(v)
#
#     # expV = np.vectorize(math.exp)
#
#     # v = np.add(np.log2(np.add(255,-v),v)-1)
#
#     v = np.add(np.square(np.add(255,-v)) * 0.07,-v)
#
#     print(v)
#     hsv[:,:,2] = v
#     img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
#
#     return img

def fFTransformFiltering(img,ref=None,deNoiseAmount=23,mode='auto'):

    img_b = img[:,:,0]
    img_g = img[:,:,1]
    img_r = img[:,:,2]

    if mode =='auto':
        deNoiseAmount = 23

    def convertdft(img):
        f = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]) + 1)

        rows, cols = img.shape
        crow, ccol = int(rows / 2), int(cols / 2)

        print(deNoiseAmount)
        mask = np.ones((rows, cols, 2), np.uint8)
        r = int(np.max(img.shape) * (int(deNoiseAmount)/100))
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 >= r * r
        mask[mask_area] = 0

        fshift_masked = fshift * mask
        magnitude_spectrum_masked = 20 * np.log(cv2.magnitude(fshift_masked[:, :, 0], fshift_masked[:, :, 1]) + 1)
        fishift_masked = np.fft.ifftshift(fshift_masked)
        img_back = cv2.idft(fishift_masked)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        img_back = cv2.normalize(img_back,None,0,1,cv2.NORM_MINMAX)
        # cv.imshow('test',img_back)

        # plt.subplot(141), plt.imshow(img, cmap='gray')
        # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        #
        # plt.subplot(142), plt.imshow(magnitude_spectrum, cmap='gray')
        # plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
        #
        # plt.subplot(143), plt.imshow(magnitude_spectrum_masked, cmap='gray')
        # plt.title('Magnitude Spectrum masked'), plt.xticks([]), plt.yticks([])
        #
        # plt.subplot(144), plt.imshow(img_back, cmap='gray')
        # plt.title('img_back'), plt.xticks([]), plt.yticks([])
        #
        # plt.show()

        return img_back

    deNoise_imgb = convertdft(img_b)
    deNoise_imgg = convertdft(img_g)
    deNoise_imgr = convertdft(img_r)

    deNoise_img = cv2.merge((deNoise_imgb,deNoise_imgg,deNoise_imgr))

    # smoothed = cv2.GaussianBlur(deNoise_img, (9, 9), 10)
    # deNoise_img = cv2.addWeighted(deNoise_img, 1, smoothed, -0.1, 0)

    # cv2.imshow('final',deNoise_img)
    # cv2.waitKey(10000)

    return deNoise_img

def matchHistogram(img,ref):
    img = cv2.cvtColor(img*255,cv2.COLOR_RGB2BGR)
    # ref = cv2.cvtColor(ref,cv2.COLOR_RGB2BGR)

    multi = True if img.shape[-1] > 1 else False
    matched = exposure.match_histograms(img, ref, multichannel=multi)

    matched = cv2.cvtColor(matched,cv2.COLOR_RGB2BGR)
    matched = matched/255

    return matched


def computeApporiateSize(ip_size):
    #256x256 = 4 64x64 size image
    #192x192 = 3 64x64 size image
    #128x128 = 2 64x64 size image
    #64x64


    final_size_input = (64,64,1)
    max_dim_size = np.max(ip_size)
    # if max_dim_size-1024 > -32:
    #     final_size_input = (1024,1024,16)
    # elif max_dim_size-960 > -32:
    #     final_size_input = (960,960,15)
    # elif max_dim_size-896 > -32:
    #     final_size_input = (896,896,14)
    # elif max_dim_size-832 > -32:
    #     final_size_input = (832,832,13)
    # elif max_dim_size-768 > -32:
    #     final_size_input = (768,768,12)
    # elif max_dim_size-704 > -32:
    #     final_size_input = (704,704,11)
    # elif max_dim_size-640 > -32:
    #     final_size_input = (640,640,10)
    # elif max_dim_size-576 > -32:
    #     final_size_input = (576,576,9)
    if max_dim_size-512 > -32:
        final_size_input = (512,512,8)
    elif max_dim_size-448 > -32:
        final_size_input = (448,448,7)
    elif max_dim_size-384 > -32:
        final_size_input = (384,384,6)
    elif max_dim_size-320 > -32:
        final_size_input = (320,320,5)
    elif max_dim_size-256 > -32:
        final_size_input = (256,256,4)
    elif max_dim_size-192 > -32:
        final_size_input = (192,192,3)
    elif max_dim_size-128 > -32:
        final_size_input = (128,128,2)

    return final_size_input

def autoPadInput(ip):
    close_size_input = computeApporiateSize(ip.shape)
    pad_amount_axis0 = 0
    pad_amount_axis1 = 0

    size_max_ix = np.argmax(ip.shape)

    match(size_max_ix):
        case 1:
            if ip.shape[1] < close_size_input[1]:
                pad_amount_axis1 = abs(ip.shape[1]-close_size_input[1])
                ip = np.pad(ip,((0,0),(0,pad_amount_axis1),(0,0)),constant_values=1)
            elif ip.shape[1] > close_size_input[1] and np.max(ip.shape) == ip.shape[1]:
                ratio_for_scaling = close_size_input[1]/float(ip.shape[1])
                dim = (close_size_input[1],int(ip.shape[0]*ratio_for_scaling))
                ip = cv2.resize(ip,dim)
        case 0:
            if ip.shape[0] < close_size_input[0]:
                pad_amount_axis0 = abs(ip.shape[0]-close_size_input[0])
                ip = np.pad(ip,((0,pad_amount_axis0),(0,0),(0,0)),constant_values=1)
            elif ip.shape[0] > close_size_input[0] and np.max(ip.shape) == ip.shape[0]:
                ratio_for_scaling = close_size_input[0]/float(ip.shape[0])
                dim = (int(ip.shape[1]*ratio_for_scaling),close_size_input[0])
                ip = cv2.resize(ip,dim)

    if ip.shape[1] < close_size_input[1]:
        pad_amount_axis1 = abs(ip.shape[1]-close_size_input[1])
        ip = np.pad(ip,((0,0),(0,pad_amount_axis1),(0,0)),constant_values=1)
    elif ip.shape[1] > close_size_input[1] and np.max(ip.shape) == ip.shape[1]:
        ratio_for_scaling = close_size_input[1]/float(ip.shape[1])
        dim = (close_size_input[1],int(ip.shape[0]*ratio_for_scaling))
        ip = cv2.resize(ip,dim)
    if ip.shape[0] < close_size_input[0]:
        pad_amount_axis0 = abs(ip.shape[0]-close_size_input[0])
        ip = np.pad(ip,((0,pad_amount_axis0),(0,0),(0,0)),constant_values=1)
    elif ip.shape[0] > close_size_input[0] and np.max(ip.shape) == ip.shape[0]:
        ratio_for_scaling = close_size_input[0]/float(ip.shape[0])
        dim = (int(ip.shape[1]*ratio_for_scaling),close_size_input[0])
        ip = cv2.resize(ip,dim)

    pad_info = (pad_amount_axis0,pad_amount_axis1,close_size_input)
    return ip,pad_info

def autoCroppedOutput(op,pad_info):
    target_size_pad_axis0 = 4*pad_info[0]
    target_size_pad_axis1 = 4*pad_info[1]
    op = op[0:op.shape[0]-target_size_pad_axis0,0:op.shape[1]-target_size_pad_axis1,:]
    return op

def predictedLrImage(ip, h5,denoiseAmount,mode):
    # ip = cv2.cvtColor(ip, cv2.COLOR_BGR2RGB)
    generator = load_model('gen_e_12 23.80226371314068 25000 [0.59059704 2.49955556].h5', compile=False, custom_objects={"SymmetricPadding2D": SymmetricPadding2D})

    # increased_ip = increase_brightness(ip,value=40)
    # increased_ip = adjust_shadow(ip)
    increased_ip = ip
    output = np.zeros(ip.shape)
    increased_ip = cv2.normalize(increased_ip,output,30,255,cv2.NORM_MINMAX)
    # increased_ip = adjust_brightness(ip,value=1.2)
    increased_ip = cv2.cvtColor(increased_ip,cv2.COLOR_BGR2RGB)
    increased_ip = increased_ip/float(255)

    # cv2.imshow('test',increased_ip)
    # cv2.waitKey(1000)

    print('input size' + str(ip.shape))
    padded_ip,pad_info = autoPadInput(increased_ip)
    # cv2.imshow('test',padded_ip)
    fake_imgs_rows = []
    if pad_info[2][2] != 1:
        lr_sub_images = [np.vsplit(x, pad_info[2][2]) for x in np.hsplit(padded_ip, pad_info[2][2])]
        lr_sub_images = np.vstack(lr_sub_images)

        with tf.device("cpu:0"): fake_sub_imgs =  generator.predict_on_batch(lr_sub_images)

        fake_imgs_row = fake_sub_imgs[0]
        for i in range(1,pad_info[2][2]*pad_info[2][2]+1,1):
            if i % pad_info[2][2] == 0:
                fake_imgs_rows.append(fake_imgs_row)
                if i != pad_info[2][2]*pad_info[2][2]: fake_imgs_row = fake_sub_imgs[i]
                continue
            fake_imgs_row = np.concatenate((fake_imgs_row,fake_sub_imgs[i]), axis=0)

        gen_image = np.concatenate(tuple(fake_imgs_rows), axis=1)
        gen_image = autoCroppedOutput(gen_image, pad_info)
    else:
        padded_ip = np.expand_dims(padded_ip,axis=0)
        print(padded_ip.shape)
        gen_image = generator.predict(padded_ip)

    # gen_image = gen_image*255
    # gen_image = cv2.cvtColor(gen_image,cv2.COLOR_RGB2BGR)
    # gen_image = decrease_brightness(gen_image,value=50)
    # gen_image = increase_brightness(gen_image,value=10)
    # gen_image = cv2.cvtColor(gen_image,cv2.COLOR_BGR2RGB)
    # gen_image = gen_image/255

    # gen_image = gen_image*255
    # gen_image = cv2.cvtColor(gen_image,cv2.COLOR_RGB2BGR)
    # gen_image = adjust_brightness(gen_image)
    # gen_image = cv2.cvtColor(gen_image,cv2.COLOR_BGR2RGB)
    # gen_image = gen_image/255

    output = np.zeros(gen_image.shape)
    gen_image = cv2.normalize(gen_image*255,output,0,255,cv2.NORM_MINMAX)
    temp_gen_image = cv2.cvtColor(gen_image,cv2.COLOR_RGB2BGR)
    gen_image = gen_image/255

    gen_image = matchHistogram(gen_image,ip)
    gen_image = gen_image*255
    gen_image = cv2.cvtColor(gen_image,cv2.COLOR_RGB2BGR)
    cv2.imwrite('unDenoiseSuperres.jpg',gen_image)

    # gen_image = cv2.fastNlMeansDenoisingColored(cv2.imread('unDenoiseSuperres.jpg'),None,4,4,7,31)
    if mode != 'None': gen_image_denoised = fFTransformFiltering(temp_gen_image,1,denoiseAmount,mode)
    else:   gen_image_denoised = temp_gen_image
    gen_image_denoised = cv2.cvtColor(gen_image_denoised,cv2.COLOR_BGR2RGB)
    gen_image_denoised = matchHistogram(gen_image_denoised,ip)
    gen_image_denoised = cv2.cvtColor(gen_image_denoised,cv2.COLOR_RGB2BGR)

    cv2.imwrite('DenoiseSuperres.jpg',gen_image_denoised*255)

    # cv2.imshow('testop',gen_image)
    # cv2.waitKey(100)
    # ip = cv2.cvtColor(ip,cv2.COLOR_BGR2RGB)

    # ip = cv2.cvtColor(ip,cv2.COLOR_BGR2RGB)
    # plt.figure(figsize=(16,8))
    # plt.subplot(231)
    # plt.title('LR Image')
    # # ip = ip.reshape((1,256,256,3))
    # pltip = np.expand_dims(ip, axis=0)
    # plt.imshow(pltip[0,:,:,:])
    # plt.subplot(232)
    # plt.title('Superresolution')
    # gen_image = np.expand_dims(gen_image, axis=0)
    # plt.imshow(gen_image[0,:,:,:])
    # plt.subplot(233)
    # plt.title('Bilinear algorithm')
    # Bilinear = cv2.resize(ip,(int(ip.shape[1]*4),int(ip.shape[0]*4)), interpolation=cv2.INTER_CUBIC)
    # Bilinear = np.expand_dims(Bilinear, axis=0)
    # plt.imshow(Bilinear[0, :, :, :])
    #
    # plt.show()

def predict(lrip_url,h5,denoiseAmount,mode):
    predictedLrImage(np.array(cv2.imread(lrip_url)),h5,denoiseAmount,mode)


# [x1,x2] = [lr_test,hr_test]
#

#One image URL
# src_image = cv2.imread('7cdb08db873944671d28logo.jpg')
# src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
# src_image = src_image.reshape((1,64,64,3))
# src_image = src_image/255
#
#
# tar_image = cv2.imread('7cdb08db873944671d28logohr.jpg')
# tar_image = cv2.cvtColor(tar_image, cv2.COLOR_BGR2RGB)
# tar_image = tar_image.reshape((1,256,256,3))
# tar_image = tar_image/255

#
# Index image from trainning and testing



# while True:
#     if cv2.waitKey(1) == ord('q'):
#         break
#     ix = randint(0, 800,1)
#     print(str(int(ix)))
#     src_image = cv2.imread("lr_images256/" + 'im' + str(int(ix)) +'.jpg')
#     src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
#     src_image = src_image/255
#     # src_image = src_image.reshape((1,256,256,3))
#     tar_image = cv2.imread("hr_images1024/" + 'im' + str(int(ix)) +'.jpg')
#     tar_image = cv2.cvtColor(tar_image, cv2.COLOR_BGR2RGB)
#     tar_image = tar_image/255
#     tar_image = tar_image.reshape((1,1024,1024,3))
#
#     #Spliting 256x256 image to 16 64x64 images
#     lr_sub_images = [np.vsplit(x, 4) for x in np.hsplit(src_image, 4)]
#     #Stack the array to a tensor of size (16,64,64,3) to suit the batch input
#     lr_sub_images = np.vstack(lr_sub_images)
#
#     #Using cpu to avoid OOMemory err on GPU when predict large tensor
#     with tf.device("cpu:0"): fake_sub_imgs =  generator.predict_on_batch(lr_sub_images)
#
#     #Stitching the 16 images predicted 256x256 images together to form a 1024x1024 size image
#     fake_imgs_row1 = np.concatenate((fake_sub_imgs[0],fake_sub_imgs[4],fake_sub_imgs[8],fake_sub_imgs[12]),axis=1)
#     fake_imgs_row2 = np.concatenate((fake_sub_imgs[1], fake_sub_imgs[5], fake_sub_imgs[9], fake_sub_imgs[13]),axis=1)
#     fake_imgs_row3 = np.concatenate((fake_sub_imgs[2], fake_sub_imgs[6], fake_sub_imgs[10], fake_sub_imgs[14]),axis=1)
#     fake_imgs_row4 = np.concatenate((fake_sub_imgs[3], fake_sub_imgs[7], fake_sub_imgs[11], fake_sub_imgs[15]),axis=1)
#     #Stitch by column
#
#     #Stitch by row
#     gen_image = np.concatenate((fake_imgs_row1,fake_imgs_row2,fake_imgs_row3,fake_imgs_row4),axis=0)
#
#     #Plot the result
#     plt.figure(figsize=(16,8))
#     plt.subplot(231)
#     plt.title('LR Image')
#     src_image = src_image.reshape((1,256,256,3))
#     plt.imshow(src_image[0,:,:,:])
#     plt.subplot(232)
#     plt.title('Superresolution')
#     gen_image = gen_image.reshape((1,1024,1024,3))
#     plt.imshow(gen_image[0,:,:,:])
#     plt.subplot(233)
#     plt.title('Orig HR Image')
#     plt.imshow(tar_image[0,:,:,:])
#
#
#     plt.show()
