import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from model import ResNet18Hotdog, HotdogClassifier


class GradCAMVisualizer: # I just used most of the reference code from the pytorch-grad-cam library
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):

        self.device = device
        self.model = self._load_model(model_path)
        self.target_layers = [self.model.model.model.layer4[-1]]  # Last conv layer of ResNet18
        self.cam = GradCAM(model=self.model, target_layers=self.target_layers)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.inv_transform = transforms.Compose([
            transforms.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225]),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
            transforms.ToPILImage()
        ])
        
        self.class_names = ['Hotdog', 'Not Hotdog']  # hotdog=0, nothotdog=1 
    
    def _load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        
        backbone = ResNet18Hotdog(pretrained=False)
        model = HotdogClassifier(backbone)
        
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        model.to(self.device)
        
        return model
    
    def preprocess_image(self, image_path):

        image = Image.open(image_path).convert('RGB')
        original_image = image.copy()
        
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        return input_tensor, original_image
    
    def _get_true_label_from_path(self, image_path):
        """Extract true label from image path based on directory structure"""
        import os
        parent_dir = os.path.basename(os.path.dirname(image_path))
        
        if parent_dir == 'hotdog':
            return 0
        elif parent_dir == 'nothotdog':
            return 1

    
    def predict_and_visualize(self, image_path, save_path=None):

        input_tensor, original_image = self.preprocess_image(image_path)
        
        true_label = self._get_true_label_from_path(image_path)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        targets = [ClassifierOutputTarget(predicted_class)]
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        
        input_image = input_tensor.squeeze(0).cpu()
        input_image = self.inv_transform(input_image)
        input_image = np.array(input_image) / 255.0
        
        visualization = show_cam_on_image(input_image, grayscale_cam[0], use_rgb=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image with true label
        axes[0].imshow(original_image)
        true_label_name = self.class_names[true_label] if true_label is not None else "Unknown"
        axes[0].set_title(f'Original Image\nTrue Label: {true_label_name}')
        axes[0].axis('off')
        
        # Grad-CAM heatmap
        axes[1].imshow(grayscale_cam[0], cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        
        # Overlay with prediction and correctness
        predicted_name = self.class_names[predicted_class]
        is_correct = "✓" if true_label == predicted_class else "✗"
        color = "green" if true_label == predicted_class else "red"
        
        axes[2].imshow(visualization)
        axes[2].set_title(f'Prediction: {predicted_name} {is_correct}\nConfidence: {confidence:.3f}', color=color)
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
        
        return {
            'predicted_class': predicted_class,
            'class_name': self.class_names[predicted_class],
            'confidence': confidence,
            'probabilities': probabilities[0].cpu().numpy(),
            'visualization': visualization,
            'true_label': true_label,
            'true_label_name': true_label_name,
            'is_correct': true_label == predicted_class
        }
    
    # def visualize_batch(self, image_paths, save_dir=None):
    #     """
    #     Visualize multiple images in a batch
        
    #     Args:
    #         image_paths: List of image paths
    #         save_dir: Optional directory to save visualizations
    #     """
    #     results = []
        
    #     for i, image_path in enumerate(image_paths):
    #         print(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
            
    #         save_path = None
    #         if save_dir:
    #             import os
    #             os.makedirs(save_dir, exist_ok=True)
    #             filename = f"gradcam_{i+1}.png"
    #             save_path = os.path.join(save_dir, filename)
            
    #         result = self.predict_and_visualize(image_path, save_path)
    #         results.append(result)
        
    #     return results

