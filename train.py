import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch

from model import ResNet18Hotdog, HotdogClassifier
from datamodule import HotdogDataModule


def main():
    # Set random seeds for reproducibility
    pl.seed_everything(42)
    
    # Initialize data module
    dm = HotdogDataModule(
        data_dir="hotdog_nothotdog",
        batch_size=64,
        num_workers=4,
        val_split=0.2,
        image_size=224
    )
    
    # Print dataset information
    print("=== Dataset Information ===")
    info = dm.get_dataset_info()
    for key, value in info.items():
        print(f"{key}: {value}")
    print()
    
    # Initialize model
    backbone = ResNet18Hotdog()
    model = HotdogClassifier(model=backbone, lr=0.001)
    
    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        mode='max',
        save_top_k=1,
        save_last=True,
        filename='hotdog-{epoch:02d}-{val_acc:.3f}',
        verbose=True
    )
    
    early_stopping = EarlyStopping(
        monitor='val_acc',
        mode='max',
        patience=5,
        verbose=True
    )
    
    # Initialize trainer (using default console logging)
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator='auto',  # Automatically use GPU if available
        devices='auto',
        callbacks=[checkpoint_callback, early_stopping],
        log_every_n_steps=10,
        val_check_interval=1.0,  # Validate after each epoch
        enable_progress_bar=True,  # Show progress bar
        enable_model_summary=True,  # Show model summary
    )
    
    print("=== Starting Training ===")
    # Train the model
    trainer.fit(model, dm)
    
    print("=== Training Complete ===")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    print(f"Best validation accuracy: {checkpoint_callback.best_model_score:.3f}")
    
    # Test the model
    print("\n=== Testing Model ===")
    trainer.test(model, dm)
    
    print("\n=== Training Summary ===")
    print(f"Total epochs trained: {trainer.current_epoch + 1}")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
