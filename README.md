# Keras SRGAN Reimplementation with modified Generator
**Keras Re-Implementation of SRGAN with symmetric padding for a better merged final image (Anysize input with 4x upscaled)**


AI-upscale application with user-interface for anysize input image with 4x upscaled output based on SRGAN with modified-Generator.
Added Symmetric padding for a seamless final merged image, add more conv layers as well as more residual blocks, lay out general workflow for anysize input.


**Modified-Generator**
![image](https://user-images.githubusercontent.com/82665400/206855054-fbce99c8-4ad0-47ee-af15-33ce8958264b.png)

**Original Discriminator**
![image](https://user-images.githubusercontent.com/82665400/206855129-b72f86ca-7e54-4a6e-bec9-442e66f6ca73.png)

**Pipeline**
![image](https://user-images.githubusercontent.com/82665400/206857747-d10e8a11-5b83-4cf0-8f61-74e1910bbd5f.png)


**NOTE**: Only support for size up to max dim of **512** (Since perdiction is using predict_on_batch with max tensor size of (16,64,64,3) (that why we are using CPU for prediction to avoid OORam)) if you are using a more powerful CPU or planning to reimplement the code with (1,16,16,3) size input tensor per splited sub image and vstack them later, you can defined the maximum size input by changing the func computeApporiateSize

![image](https://user-images.githubusercontent.com/82665400/206857420-a514481e-5a0a-42d0-98ea-e8819efbae51.png)


