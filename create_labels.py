from torchvision.models import ResNet50_Weights
import json

def create_imagenet_labels():
    # Get the ImageNet class mapping
    weights = ResNet50_Weights.IMAGENET1K_V1
    class_labels = weights.meta["categories"]
    
    # Create dictionary with all 1000 classes
    label_dict = {}
    for idx, label in enumerate(class_labels):
        label_dict[str(idx)] = label
    
    # Save to file
    with open('imagenet_classes.json', 'w') as f:
        json.dump(label_dict, f, indent=4)
    
    print(f"Created labels file with {len(label_dict)} classes")

if __name__ == "__main__":
    create_imagenet_labels() 