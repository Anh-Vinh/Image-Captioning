# Image Captioning with InceptionNet and LSTM (PyTorch)
This repository implements an image captioning model that generates natural language descriptions from images. The architecture uses InceptionNet for feature extraction and an LSTM-based decoder for generating captions. The model design is based on a YouTube tutorial, with custom modifications in the dataset pipeline and training loop to suit specific training needs.

# Model Architecture
- Encoder: InceptionNet (pretrained on ImageNet, final layer removed)
- Decoder: LSTM with embedding and fully connected layers
- Loss Function: CrossEntropyLoss
- Optimizer: Adam

# Data
The Flickr8k dataset from Kaggle contains 8,000 images with 5 captions per image.
Data Split: The dataset is split using an 8:1:1 ratio:
- Training set: 6,400 images
- Validation set: 800 images
- Test set: 800 images

# Training
- Learning rate: 1e-3
- Epoch: 15
- Batch size: 32
- Embed size: 256
- Hidden size: 256
- Num layers: 1
- Vocab threshold: 5

--num_workers:
# Result


# References:
- Tutorial video: https://www.youtube.com/watch?v=y2BaTt1fxJU
- Kaggle dataset: https://www.kaggle.com/datasets/adityajn105/flickr8k
