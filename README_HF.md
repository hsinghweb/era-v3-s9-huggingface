---
title: ResNet50 Image Classifier
emoji: 🖼️
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: 1.22.0
app_file: app.py
pinned: false
---

# ResNet50 Image Classifier

This Streamlit application uses a ResNet50 model trained on the ImageNet-1K dataset to classify images into 1000 different categories.

## How to Use

1. Click the "Choose an image..." button or drag and drop an image
2. The model will automatically process your image
3. View the top 5 predictions with their confidence scores

## Model Details

- **Architecture**: ResNet50
- **Dataset**: ImageNet-1K
- **Input Size**: 224x224 pixels
- **Number of Classes**: 1000

## Example Predictions

The model can identify various objects, animals, and scenes, including:
- Common animals (dogs, cats, birds)
- Everyday objects
- Vehicles
- Natural scenes
- And many more!

## Technical Details

- Built with PyTorch and Streamlit
- Uses standard ImageNet preprocessing
- Runs inference on CPU
- Displays confidence scores as progress bars

## Note

For best results, use clear, well-lit images with a single main subject. 