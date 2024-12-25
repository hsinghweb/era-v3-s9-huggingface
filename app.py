import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import json

# Load ImageNet class labels
with open('imagenet_classes.json') as f:
    labels = json.load(f)

def load_model():
    model = resnet50()
    model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

def process_image(image):
    # Define the same transforms used during training
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = transform(image).unsqueeze(0)
    return image

def get_prediction(model, image):
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top5_prob, top5_catid = torch.topk(probabilities, 5)
    return top5_prob, top5_catid

def main():
    st.title("Image Classification with ResNet50")
    st.write("Upload an image and the model will predict its category")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Load model
        model = load_model()
        
        # Process image and get prediction
        processed_image = process_image(image)
        top5_prob, top5_catid = get_prediction(model, processed_image)
        
        # Display predictions
        st.subheader("Predictions:")
        for i in range(5):
            probability = top5_prob[i].item() * 100
            category = labels[str(top5_catid[i].item())]
            st.write(f"{category}: {probability:.2f}%")
            st.progress(probability/100)

if __name__ == "__main__":
    main() 