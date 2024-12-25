import torch
import os
from torchvision.models import resnet50, ResNet50_Weights

def download_pretrained_model():
    try:
        # Load ResNet50 model with the best available weights
        print("Downloading ResNet50 model with ImageNet-1K weights...")
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.eval()
        
        # Save the model with safe loading
        print("Saving model to best_model.pth...")
        torch.save(model.state_dict(), 'best_model.pth', _use_new_zipfile_serialization=True)
        
        # Verify the file exists
        if os.path.exists('best_model.pth'):
            model_size = os.path.getsize('best_model.pth') / (1024 * 1024)  # Size in MB
            print(f"Model saved successfully! Size: {model_size:.2f} MB")
        else:
            print("Error: Model file was not created")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    download_pretrained_model() 