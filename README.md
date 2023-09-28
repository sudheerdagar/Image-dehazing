# Real-time Dehazing

This repository contains a real-time dehazing model.

## Model Details
- The model is currently based on an encoder-decoder architecture.
- It is trained on a dataset of 55 images.
- The model's purpose is to remove haze and enhance image visibility.

## Future Improvements

We have plans to make several improvements to this dehazing model:

1. **Attention Module:** We intend to introduce an attention mechanism to make the model more effective in identifying and removing haze from important regions of the image.

2. **Reduced Layers:** We plan to optimize the model architecture by reducing the number of layers, making it more efficient without compromising performance.

3. **Input and Output Image Size Consideration:** To ensure that the input image size and output image size are almost the same, we will refine the model's architecture and preprocessing steps. This will help maintain image consistency and improve the overall dehazing process.

4. **Indoor Image Generation with Cyclic GAN:** To expand the training dataset, we will use a CycleGAN to generate more indoor images. This will enhance the model's ability to handle a wider range of scenarios.

5. **Enhanced Preprocessing for Real-time Video:** To achieve real-time processing for videos, we will streamline the preprocessing steps, optimizing them for speed without sacrificing dehazing quality. This will improve the model's suitability for video applications, ensuring a good frame rate.

## Sample Images

Here are some trained images for illustration:

| Real Image (p1.png) | Fogged Image (p2.png) | Dehazed Image (p3.png) |
|---------------------|-----------------------|------------------------|
| ![Real Image](/Images/p1.png) | ![Fogged Image](/Images/p2.png) | ![Dehazed Image](/Images/p3.png) |


## Example Usage

You can use this model to dehaze your own images. For example, let's take the following images:

- Ground Image (fog2.jpg)
- Generated Image (p4.png)

| Ground Image (fog2.jpg) | Generated Image (p4.png) |
|-------------------------|---------------------------|
| ![Ground Image](/Images/fog2.jpg) | ![Generated Image](/Images/p4.png) |

To dehaze an image, simply pass it through the model.

```python
#To generate an image
import cv2
import numpy as np
import torch

# Load and preprocess the image
img = cv2.imread("fog2.jpg")
img = cv2.resize(img, (256, 256))
img = np.array(img)
img = np.array([img])
img = np.transpose(img,(0,3,1,2))
img=img.reshape(-1,1,256,256)  # Convert to grayscale
print(img.shape)
img = img / 255.0  # Normalize the grayscale image

# Transfer the tensor to the GPU if available
if torch.cuda.is_available():
    img = torch.FloatTensor(img).cuda()


train_hazy_loader = torch.utils.data.DataLoader(dataset=img,batch_size=batch_size,shuffle=False)

dehazed_output=[]
for train_hazy in tqdm(train_hazy_loader):
    hazy_image = Variable(train_hazy).cuda()

    encoder_op = encoder(hazy_image)
    output = decoder(encoder_op)

    output=output.cpu()
    output=output.detach()
    dehazed_output.append(output)

X_dehazed=dehazed_output
X_dehazed=torch.stack(X_dehazed)
print(X_dehazed.size())
X_dehazed=X_dehazed.view(-1,1,256,256)
print(X_dehazed.size())
X_dehazed=X_dehazed.view(-1,3,256,256)
print(X_dehazed.size())
X_dehazed=X_dehazed.permute(0,2,3,1)
print(X_dehazed.shape)
for i in X_dehazed:
  plt.imshow(i)
