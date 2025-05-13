# Image Captioning with InceptionNet and LSTM (PyTorch)
This repository implements an image captioning model that generates natural language descriptions from images. The architecture uses InceptionNet for feature extraction and an LSTM-based decoder for generating captions. The model design is based on a YouTube tutorial, with custom modifications in the dataset pipeline and training loop to suit specific training needs.

# Model Architecture
- Encoder: InceptionNet (pretrained on ImageNet, final layer removed)
- Decoder: LSTM with embedding and fully connected layers
- Loss Function: CrossEntropyLoss
- Optimizer: Adam

# Data
The Flickr8k dataset from Kaggle contains 8,000 images with 5 captions per image. The dataset is split using an 8:1:1 ratio:
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

# Result
- Training loss: 2.2
- Validating loss: 2.4
- Testing BLEU score: 0.1
The model is still relatively weak overall. It can identify the main subject in images, especially more visually distinct objects like dogs, and generate diverse captions. However, it struggles with complex scenes, likely due to the simplicity of sentence structures in the Flickr8k dataset.
Some common issues observed:
- Incorrect subject gender in human-related captions
- Limited ability to describe interactions or detailed context
- Repetitive or generic captions in some cases
![plot](https://github.com/user-attachments/assets/d3cb4e2b-8c76-4a06-acf2-729f3be1a822)
![plot (1)](https://github.com/user-attachments/assets/9a1e004d-97b4-4e5c-b845-591e479fc4a7)
![plot (2)](https://github.com/user-attachments/assets/bb47b4c9-1faf-4260-bbc9-076f45404f4f)

# References:
- Tutorial video: https://www.youtube.com/watch?v=y2BaTt1fxJU
- Kaggle dataset: https://www.kaggle.com/datasets/adityajn105/flickr8k
