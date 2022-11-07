import os
import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import layers, Model
from sklearn.model_selection import train_test_split
import numpy as np
from keras import Model
from keras.layers import Conv2D, PReLU, BatchNormalization,Flatten
from keras.layers import UpSampling2D, LeakyReLU, Dense, Input, add
from tqdm import tqdm #progess bar
from random import randint
import tensorflow as tf

#Self defined custom padding layer
from CustomPadding import SymmetricPadding2D

def res_block(ip):

    #Using padding before convolution to avoid losing data at the rear of the image
    res_model = SymmetricPadding2D(1)(ip)
    res_model = SymmetricPadding2D(1)(res_model)
    res_model = Conv2D(64,(5,5), padding="valid")(res_model)
    res_model = BatchNormalization(momentum = 0.5)(res_model)


    #Different filter size will convolute to different size tensor, so different number of padding layer
    #inorder to keep the tensor the same size
    res_model = SymmetricPadding2D(1)(res_model)
    res_model = Conv2D(64, (3,3), padding = "valid")(res_model)
    res_model = BatchNormalization(momentum = 0.5)(res_model)
    res_model = PReLU(shared_axes = [1,2])(res_model)

    #'valid' padding in conv2D mean no padding
    res_model = SymmetricPadding2D(1)(res_model)
    res_model = Conv2D(64, (3,3), padding = "valid")(res_model)
    res_model = BatchNormalization(momentum = 0.5)(res_model)

    return add([ip, res_model])

def upscale_block(ip):

    up_model = SymmetricPadding2D(1)(ip)
    up_model = Conv2D(256, (3,3), padding="valid")(up_model)
    up_model = PReLU(shared_axes=[1,2])(up_model)

    up_model = SymmetricPadding2D(1)(up_model)
    up_model = Conv2D(256, (3,3), padding="valid")(up_model)
    up_model = BatchNormalization(momentum=0.5)(up_model)
    up_model = UpSampling2D(size=2)(up_model)
    up_model = PReLU(shared_axes=[1,2])(up_model)

    return up_model

def create_gen(gen_ip, num_res_block):

    #The reason we use Symmetric padding over zero padding is because we don't want junk data
    #on the edge of the image, cause we still want to stitch the image together
    #Destroying the data at the edge via zero padding will result in ugly stitch image with border
    layers = SymmetricPadding2D(1)(gen_ip)
    layers = SymmetricPadding2D(1)(layers)
    layers = SymmetricPadding2D(1)(layers)
    layers = SymmetricPadding2D(1)(layers)
    layers = Conv2D(64, (9,9),padding = "valid")(layers)
    layers = PReLU(shared_axes=[1,2])(layers)

    layers = SymmetricPadding2D(1)(layers)
    layers = SymmetricPadding2D(1)(layers)
    layers = Conv2D(64, (5,5),padding = "valid")(layers)
    layers = PReLU(shared_axes=[1,2])(layers)

    #Temp use for elementwise sum, for skip connection
    temp = layers

    #Residual blocks
    for i in range(num_res_block):
        layers = res_block(layers)

    layers = SymmetricPadding2D(1)(layers)
    layers = Conv2D(64, (3,3), padding="valid")(layers)
    layers = BatchNormalization(momentum=0.5)(layers)

    #Add the temp from previous layer for skip connection
    layers = add([layers,temp])

    #Upsacle the image by 2x for each block
    layers = upscale_block(layers)
    layers = upscale_block(layers)

    #outputShape = RoundedDown((n+2*paddingsize-filtersSize)/stride)+1
    layers = SymmetricPadding2D(1)(layers)
    layers = SymmetricPadding2D(1)(layers)
    layers = SymmetricPadding2D(1)(layers)
    layers = SymmetricPadding2D(1)(layers)
    op = Conv2D(3, (9,9), padding="valid")(layers)

    return Model(inputs=gen_ip, outputs=op)

def discriminator_block(ip, filters, strides=1, bn=True):
    disc_model = Conv2D(filters, (3,3) ,strides = strides, padding="same")(ip)

    if bn:
        #BatchNormalization for reducing the variance and get the mean to 0
        #meant to speed up the model training time
        disc_model = BatchNormalization( momentum=0.8)(disc_model)

    #Using LeakyRelu so the model can learn even for negative input
    disc_model = LeakyReLU( alpha=0.3)(disc_model)

    return disc_model

def create_disc(disc_ip):
    df = 32

    d1 = discriminator_block(disc_ip, df , bn = False)
    d2 = discriminator_block(d1, df, strides=2)
    d3 = discriminator_block(d2, df*2)
    d4 = discriminator_block(d3, df*2, strides = 2)
    d5 = discriminator_block(d4, df*4)
    d6 = discriminator_block(d5, df*4, strides=2)
    d7 = discriminator_block(d6, df*8)
    d8 = discriminator_block(d7, df*8, strides=2)

    #Fully connected layer to decide the image input is from the generator or real
    d8_5 = Flatten()(d8)
    d9 = Dense(df*16)(d8_5)
    d10 = LeakyReLU(alpha=0.3)(d9)
    validity = Dense(1, activation='sigmoid')(d10)

    return Model(disc_ip, validity)

from keras.applications.vgg19 import VGG19

def build_vgg(hr_shape):

    #Selected vgg layers for perceptual loss
    selectedLayers = [10,5,8]
    vgg = VGG19(weights="imagenet",include_top=False, input_shape=hr_shape)
    selectedOutputs = [vgg.layers[i].output for i in selectedLayers]

    return Model(inputs=vgg.inputs, outputs=selectedOutputs)

def create_comb(gen_model, disc_model, vgg, lr_ip, hr_ip):
    gen_img = gen_model(lr_ip)
    #Taking the features of the gen_img and compare with features of hr image
    gen_features = vgg(gen_img)

    disc_model.trainable = False
    validity = disc_model(gen_img)

    return Model(inputs=[lr_ip, hr_ip], outputs=[validity, gen_features])

n=25000
lr_list = os.listdir("lr_images")[:n]

# lr_images = []
# for img in lr_list:
#     img_lr = cv2.imread("lr_images/"+img)
#     img_lr = cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB)
#     lr_images.append(img_lr)

#Only loading the image url to save on Ram
lr_image_urls = []
for url in lr_list:
    lr_image_urls.append(str("lr_images/"+url))

hr_list = os.listdir("hr_images")[:n]

# hr_images = []
# for img in hr_list:
#     img_hr = cv2.imread("hr_images/"+img)
#     img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB)
#     hr_images.append(img_hr)
hr_image_urls = []
for url in lr_list:
    hr_image_urls.append(str("hr_images/"+url))

# lr_images = np.array(lr_images)
# hr_images = np.array(hr_images)

#Plot random lr and hr image
# import random
# import numpy as np
# image_number = random.randint(0,len(lr_images)-1)
# plt.figure(figsize=(12,6))
# plt.subplot(121)
# plt.imshow(np.reshape(lr_images[image_number],(32,32,3)))
# plt.subplot(122)
# plt.imshow(np.reshape(hr_images[image_number],(128,128,3)))
# plt.show()

# lr_images = lr_images/255
# hr_images = hr_images/255

lr_train, lr_test, hr_train, hr_test = train_test_split(lr_image_urls, hr_image_urls,test_size=0.01,random_state=True)

# hr_shape = (hr_train.shape[1], hr_train.shape[2], hr_train.shape[3])
# lr_shape = (lr_train.shape[1], lr_train.shape[2], lr_train.shape[3])

#Defining the input shape manually since we only load the url into the training array
hr_shape = (256,256,3)
lr_shape = (64,64,3)

lr_ip = Input(shape=lr_shape)
hr_ip = Input(shape=hr_shape)

#Creating the generator
generator = create_gen(lr_ip, num_res_block=20)
generator.summary()

#Creating the discriminator
discriminator = create_disc(hr_ip)
discriminator.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])
discriminator.summary()

#Creating the vgg for perceptual loss
vgg = build_vgg((256,256,3))
print(vgg.summary())
vgg.trainable = False

#Combning all together
gan_model = create_comb(generator,discriminator,vgg,lr_ip,hr_ip)
#Binary_crossentropy for loss of discriminator,
#mse for loss of gen_image and hr image when compare to different layer of vgg
gan_model.compile(loss=["binary_crossentropy","mse"], loss_weights=[1e-3,1],optimizer="adam")
gan_model.summary()

#Batch size will stay at 1 since dataset is varying in subject with little correlation
batch_size = 1
train_lr_batches = []
train_hr_batches = []

#Spliting the data for training on batch
for it in range(int(len(hr_train)/batch_size)):
    start_idx = it * batch_size
    end_idx = start_idx + batch_size
    train_hr_batches.append(hr_train[start_idx:end_idx])
    train_lr_batches.append(lr_train[start_idx:end_idx])

#Number of epochs
epochs = 30

for e in range(epochs):
    #Label for gen_images and real images to feed into discriminator
    fake_label = np.zeros((batch_size,1))
    real_label = np.ones((batch_size,1))

    g_losses = []
    d_losses = []

    for b in tqdm(range(len(train_hr_batches))):
        #Loading image from url
        lr_image_urls = train_lr_batches[b]
        lr_images = cv2.imread("".join(lr_image_urls))
        # print("".join(lr_image_urls))
        lr_images = np.array(lr_images)
        lr_images = cv2.cvtColor(lr_images, cv2.COLOR_BGR2RGB)
        lr_images = lr_images/255

        hr_image_urls = train_hr_batches[b]
        hr_images = cv2.imread("".join(hr_image_urls))
        hr_images = np.array(hr_images)
        hr_images = cv2.cvtColor(hr_images, cv2.COLOR_BGR2RGB)
        hr_images = hr_images/255

        # lr_sub_images = [np.vsplit(x,4) for x in np.hsplit(lr_images,4)]
        # hr_sub_images = [np.vsplit(x,4) for x in np.hsplit(hr_images,4)]
        #
        # lr_sub_images = np.vstack(lr_sub_images)
        # hr_sub_images = np.vstack(hr_sub_images)

        # for subBatch in range(0,16,2):

            # lr_sub_batch = lr_sub_images[subBatch:subBatch+2]
            # hr_sub_batch = hr_sub_images[subBatch:subBatch+2]

        #Since we don't load the image beforehand, the size of image is still (x,y,channels)
        #But we want it to have size of (n,x,y,channels) to fit the model input
        lr_images = np.expand_dims(lr_images, axis=0)
        hr_images = np.expand_dims(hr_images,axis=0)


            # fake_sub_imgs = generator.predict_on_batch(lr_sub_images)
            # print(lr_sub_images.shape)
            # with tf.device("cpu:0"): fake_imgs =  generator.predict_on_batch(lr_sub_images)

        #Generating hr images from lr
            # fake_imgs = generator.predict_on_batch(lr_sub_batch)
        fake_imgs = generator.predict_on_batch(lr_images)

        #Training discriminator
        discriminator.trainable = True

        #Getting random int to decide if fake or real image we want to train the discriminator first
        ix = randint(0,1)

            # fake_imgs_row1 = np.concatenate((fake_sub_imgs[0],fake_sub_imgs[1],fake_sub_imgs[2],fake_sub_imgs[3]),axis=1)
            # fake_imgs_row2 = np.concatenate((fake_sub_imgs[4], fake_sub_imgs[5], fake_sub_imgs[6], fake_sub_imgs[7]),axis=1)
            # fake_imgs_row3 = np.concatenate((fake_sub_imgs[8], fake_sub_imgs[9], fake_sub_imgs[10], fake_sub_imgs[11]),axis=1)
            # fake_imgs_row4 = np.concatenate((fake_sub_imgs[12], fake_sub_imgs[13], fake_sub_imgs[14], fake_sub_imgs[15]),axis=1)
            #
            # fake_imgs = np.concatenate((fake_imgs_row1,fake_imgs_row2,fake_imgs_row3,fake_imgs_row4),axis=0)

            # selected_traindata = [fake_imgs,hr_sub_images]
            # selected_traindata = [fake_imgs,hr_sub_batch]

        selected_traindata = [fake_imgs,hr_images]
        selected_trainlabel = [fake_label,real_label]

        temp = discriminator.train_on_batch(selected_traindata[ix],selected_trainlabel[ix])
        match ix:
            case 0:
                d_losses_real = discriminator.train_on_batch(selected_traindata[1], selected_trainlabel[1])
                d_losses_gen = temp
            case 1:
                d_losses_gen = discriminator.train_on_batch(selected_traindata[0], selected_trainlabel[0])
                d_losses_real = temp

        discriminator.trainable = False

        #Adding the loss of real and fake image by the generator
        d_loss = 0.5 + np.add(d_losses_gen, d_losses_real)
        # image_features = vgg.predict_on_batch(hr_sub_batch)

        #Getting hr images features and compare the L2 distance for each example and computing the MSE loss
        image_features = vgg.predict(hr_images)

        # g_loss, _, _ = gan_model.train_on_batch([lr_sub_batch, hr_sub_batch],[real_label, image_features])
        #Getting the whole gan loss of the model
        g_loss, _, _ = gan_model.train_on_batch([lr_images, hr_images], [real_label, image_features])

        d_losses.append(d_loss)
        g_losses.append(g_loss)

    g_losses = np.array(g_losses)
    d_losses = np.array(d_losses)

    #Computing the loss of gan and discriminator for
    g_loss = np.sum(g_losses, axis=0)/ len(g_losses)
    d_loss = np.sum(d_losses,axis=0) / len(d_losses)

    print("epoch:", e+1, "g_loss:", g_loss, "d_loss:", d_loss)

    generator.save("gen_e_"+str(e+1)+" "+str(g_loss)+" "+str(n)+" "+str(d_loss)+".h5")




#Imageflow
# filepath = "Treeweights-{epoch:82d}-{val_accuracy:.2f}.hdf5"
# checkpoint = ModelCheckpoint(filepath,monitor='val_accuracy',verbose=1,save_best_only=True,mode='max')
# callbacks_list = [checkpoint]
#
# aug = ImageDataGenerator(rotation_range=20, zoom_range=0.2,
#      rescale=1./255,
#     width_shift_range=0.1,
#       height_shift_range=0.1,
#   	horizontal_flip=True,
#      brightness_range=[0.5,2.5], fill_mode="nearest")
#
# aug_val = ImageDataGenerator(rescale=1./255)

#Fit_model
# vgghist=vggmodel.fit_generator(aug.flow(x_train, y_train, batch_size=64),
#                                  epochs=50,
#                                  validation_data=aug.flow(x_test,y_test,
#                                  batch_size=64),
#                                  callbacks=callbacks_list)


#Load weights
# generator = create_gen(lr_ip, num_res_block=16)
# generator.summary()
#
# discriminator = create_disc(hr_ip)
# discriminator.compile(loss="binary_crossentopy",optimizer="adam",metrics=['accuracy'])
# discriminator.summary()
#
# vgg = build_vgg((128,128,3))
# print(vgg.summary())
# vgg.trainable = False
#
# gan_model = create_comb(generator,discriminator,vgg,lr_ip,hr_ip)
# gan_model.load_weights()

