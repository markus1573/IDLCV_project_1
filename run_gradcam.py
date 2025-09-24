"""
Simple script to run Grad-CAM visualization on test images
"""

import os
import glob
import random
from gradcam_visualizer import GradCAMVisualizer


def main():
    model_path = "lightning_logs/version_17/checkpoints/hotdog-epoch=25-val_acc=0.973.ckpt"
    test_dir = "hotdog_nothotdog/test"
    output_dir = "gradcam_results"
    
    if not os.path.exists(model_path):
        print(f"Model checkpoint not found: {model_path}")
        print("Available checkpoints:")
        for root, dirs, files in os.walk("lightning_logs"):
            for file in files:
                if file.endswith('.ckpt'):
                    print(f"  {os.path.join(root, file)}")
        return
    
    print("Loading model...")
    visualizer = GradCAMVisualizer(model_path)
    
    hotdog_images = glob.glob(os.path.join(test_dir, "hotdog", "*.jpg"))
    nothotdog_images = glob.glob(os.path.join(test_dir, "nothotdog", "*.jpg"))
    
    sample_hotdog = random.sample(hotdog_images, min(3, len(hotdog_images)))
    sample_nothotdog = random.sample(nothotdog_images, min(3, len(nothotdog_images)))
    
    all_samples = sample_hotdog + sample_nothotdog
    random.shuffle(all_samples)
    

    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    for i, image_path in enumerate(all_samples):
        print(f"\nProcessing {i+1}/{len(all_samples)}: {os.path.basename(image_path)}")
        
        save_path = os.path.join(output_dir, f"gradcam_{i+1:02d}.png")
        result = visualizer.predict_and_visualize(image_path, save_path)
        results.append({
            'image_path': image_path,
            'prediction': result['class_name'],
            'confidence': result['confidence']
        })
    
    print("\n" + "="*50)
    print("GRAD-CAM VISUALIZATION SUMMARY")
    print("="*50)
    
    correct_predictions = 0
    for i, result in enumerate(results):
        filename = os.path.basename(result['image_path'])
        
        if 'true_label_name' in result:
            true_label = result['true_label_name']
            prediction = result['prediction']
            confidence = result['confidence']
            is_correct = result['is_correct']
            
            if is_correct:
                correct_predictions += 1
                status = "✓"
            else:
                status = "✗"
            
            print(f"{i+1:2d}. {filename:20s} | True: {true_label:12s} | Pred: {prediction:12s} {status} ({confidence:.3f})")

    if correct_predictions > 0:
        accuracy = correct_predictions / len(results) * 100
        print(f"\nAccuracy on sample: {correct_predictions}/{len(results)} ({accuracy:.1f}%)")
    
    print(f"Visualizations saved to: {output_dir}/")


if __name__ == "__main__":
    main()
