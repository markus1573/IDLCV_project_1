# Hotdog vs. Not-Hotdog Image Classifier

A computer vision project that classifies images as either "hotdog" or "not hotdog" using deep learning with PyTorch Lightning and transfer learning.

## 🎯 Project Overview

This project implements a binary image classifier to distinguish between hotdog and non-hotdog images. It uses a pre-trained ResNet18 model with transfer learning, fine-tuned on a custom dataset of hotdog images.

**Key Features:**
- Transfer learning with ResNet18 backbone
- PyTorch Lightning for clean, organized training code
- Data augmentation for improved generalization
- Automatic train/validation split
- Model checkpointing and early stopping
- Comprehensive logging and metrics

## 📋 Requirements

### Python Dependencies
```
torch>=1.9.0
torchvision>=0.10.0
pytorch-lightning>=1.4.0
Pillow>=8.0.0
```

### System Requirements
- Python 3.7+
- CUDA-compatible GPU (optional, but recommended for faster training)
- At least 4GB RAM
- ~500MB disk space for the dataset

## 🚀 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/markus1573/IDLCV_project_1.git
cd IDLCV_project_1
```

2. **Install dependencies:**
```bash
# Using pip
pip install torch torchvision pytorch-lightning Pillow

# Or using conda
conda install pytorch torchvision pytorch-lightning pillow -c pytorch -c conda-forge
```

## 📁 Dataset Structure

The project expects the following directory structure:
```
hotdog_nothotdog/
├── train/
│   ├── hotdog/        # Training hotdog images
│   └── nothotdog/     # Training non-hotdog images
└── test/
    ├── hotdog/        # Test hotdog images
    └── nothotdog/     # Test non-hotdog images
```

**Dataset Statistics:**
- Total images: ~3,900
- Classes: 2 (hotdog, nothotdog)
- Format: JPEG images
- Split: 80% training, 20% validation (from train folder), separate test set

## 🏃‍♂️ How to Run

### Basic Training
```bash
python train.py
```

### Training Configuration
The default configuration in `train.py` includes:
- **Batch size:** 64
- **Learning rate:** 0.001
- **Max epochs:** 20
- **Image size:** 224x224
- **Validation split:** 20%
- **Early stopping:** 5 epochs patience

### Customization
You can modify training parameters by editing the values in `train.py`:

```python
# In train.py, modify these values:
dm = HotdogDataModule(
    data_dir="hotdog_nothotdog",
    batch_size=64,           # Adjust batch size
    num_workers=4,           # Adjust for your CPU
    val_split=0.2,           # Validation split ratio
    image_size=224           # Input image size
)

model = HotdogClassifier(
    model=backbone, 
    lr=0.001                 # Learning rate
)
```

## 🏗️ Model Architecture

**Base Model:** ResNet18 (pre-trained on ImageNet)
- **Input:** 224×224×3 RGB images
- **Output:** 2 classes (hotdog, nothotdog)
- **Modification:** Final fully connected layer adapted for binary classification
- **Optimizer:** Adam
- **Loss Function:** Cross Entropy Loss

**Data Augmentation:**
- Random horizontal flip (50% probability)
- Random rotation (±15 degrees)
- Color jitter (brightness, contrast, saturation, hue)
- Normalization with ImageNet statistics

## 📊 Training Output

During training, you'll see:
1. **Dataset information** - number of training/validation/test samples
2. **Training progress** - loss and accuracy metrics per epoch
3. **Validation metrics** - validation accuracy and loss
4. **Best model checkpoint** - saved automatically
5. **Test results** - final model performance on test set

Example output:
```
=== Dataset Information ===
num_classes: 2
class_names: ['hotdog', 'nothotdog']
train_size: 2000
val_size: 500
test_size: 1409
...

=== Starting Training ===
Epoch 1/20: 100%|██████████| 32/32 [00:15<00:00, train_loss=0.234, val_acc=0.876]
...

=== Training Complete ===
Best model saved at: lightning_logs/version_0/checkpoints/hotdog-epoch=05-val_acc=0.923.ckpt
Best validation accuracy: 0.923
```

## 📂 Output Files

After training, you'll find:
- **Model checkpoints:** `lightning_logs/version_X/checkpoints/`
- **Training logs:** `lightning_logs/version_X/`
- **Best model:** Automatically saved with highest validation accuracy

## 🔧 Troubleshooting

### Common Issues

1. **CUDA out of memory:**
   ```python
   # Reduce batch size in train.py
   batch_size=32  # or 16
   ```

2. **Dataset not found:**
   ```
   FileNotFoundError: [Errno 2] No such file or directory: 'hotdog_nothotdog'
   ```
   - Ensure the `hotdog_nothotdog` folder is in the project root
   - Check that it contains `train` and `test` subfolders

3. **Import errors:**
   ```
   ModuleNotFoundError: No module named 'pytorch_lightning'
   ```
   - Install missing dependencies: `pip install pytorch-lightning`

4. **Slow training:**
   - Reduce `num_workers` if you have limited CPU cores
   - Use GPU if available (automatic detection)

### Performance Tips

- **For faster training:** Use GPU and increase batch size
- **For limited memory:** Reduce batch size and image size
- **For better accuracy:** Increase epochs and add more data augmentation

## 🎮 Usage Examples

### Quick Test Run
```bash
# Test with minimal epochs for quick validation
# Modify max_epochs=2 in train.py, then:
python train.py
```

### Custom Data Path
```python
# Modify data_dir in train.py if your data is elsewhere
dm = HotdogDataModule(
    data_dir="path/to/your/dataset",
    # ... other parameters
)
```

## 📈 Expected Performance

With the default configuration, you should expect:
- **Training time:** 10-30 minutes (depending on hardware)
- **Validation accuracy:** 85-95%
- **Test accuracy:** 80-90%

## 🤝 Contributing

Feel free to submit issues or pull requests for improvements!

## 📄 License

This project is for educational purposes.