import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch
import yaml
import argparse
from pathlib import Path

from model import ResNet18Hotdog, HotdogClassifier
from datamodule import HotdogDataModule


def load_config(config_path: str = "config.yaml"):
    """Load configuration from YAML file"""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config


def main(config_path: str = "config.yaml"):
    # Load configuration
    config = load_config(config_path)
    
    # Set random seeds for reproducibility
    pl.seed_everything(config['seed'])
    
    # Initialize data module
    dm = HotdogDataModule(
        data_dir=config['data']['data_dir'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        val_split=config['data']['val_split'],
        image_size=config['data']['image_size'],
        normalize_mean=config['data']['normalize_mean'],
        normalize_std=config['data']['normalize_std'],
        pin_memory=config['data']['pin_memory'],
        persistent_workers=config['data']['persistent_workers'],
        # Data augmentation parameters
        augmentation_enabled=config['augmentation']['enabled'],
        random_flip_p=config['augmentation']['random_horizontal_flip']['p'],
        random_rotation_degrees=config['augmentation']['random_rotation']['degrees'],
        color_jitter_brightness=config['augmentation']['color_jitter']['brightness'],
        color_jitter_contrast=config['augmentation']['color_jitter']['contrast'],
        color_jitter_saturation=config['augmentation']['color_jitter']['saturation'],
        color_jitter_hue=config['augmentation']['color_jitter']['hue']
    )
    
    # Print dataset information
    print("=== Dataset Information ===")
    info = dm.get_dataset_info()
    for key, value in info.items():
        print(f"{key}: {value}")
    print()
    
    # Initialize model
    backbone = ResNet18Hotdog(config['model']['pretrained'])
    model = HotdogClassifier(
        model=backbone, 
        lr=config['model']['lr'], 
        weight_decay=config['model']['weight_decay']
    )
        
    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor=config['callbacks']['model_checkpoint']['monitor'],
        mode=config['callbacks']['model_checkpoint']['mode'],
        save_top_k=config['callbacks']['model_checkpoint']['save_top_k'],
        save_last=config['callbacks']['model_checkpoint']['save_last'],
        filename=config['callbacks']['model_checkpoint']['filename'],
        verbose=config['callbacks']['model_checkpoint']['verbose']
    )
    
    early_stopping = EarlyStopping(
        monitor=config['callbacks']['early_stopping']['monitor'],
        mode=config['callbacks']['early_stopping']['mode'],
        patience=config['callbacks']['early_stopping']['patience'],
        verbose=config['callbacks']['early_stopping']['verbose']
    )
    
    # Initialize trainer (using default console logging)
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator=config['training']['accelerator'],
        devices=config['training']['devices'],
        callbacks=[checkpoint_callback, early_stopping],
        log_every_n_steps=config['training']['log_every_n_steps'],
        val_check_interval=config['training']['val_check_interval'],
        enable_progress_bar=config['training']['enable_progress_bar'],
        enable_model_summary=config['training']['enable_model_summary'],
    )
    
    print("=== Starting Training ===")
    # Train the model
    trainer.fit(model, dm)
    
    print("=== Training Complete ===")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    print(f"Best validation loss: {checkpoint_callback.best_model_score:.3f}")
    
    # Test the model
    print("\n=== Testing Model ===")
    trainer.test(model, dm)
    
    print("\n=== Training Summary ===")
    print(f"Total epochs trained: {trainer.current_epoch + 1}")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Hotdog/Not-Hotdog classifier")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml", 
        help="Path to config file (default: config.yaml)"
    )
    args = parser.parse_args()
    
    main(args.config)
