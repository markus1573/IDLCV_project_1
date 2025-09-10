import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from dataset import Hotdog_NotHotdog
import torch


class HotdogDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "hotdog_nothotdog",
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.2,
        image_size: int = 224,
        normalize_mean: tuple = (0.485, 0.456, 0.406),
        normalize_std: tuple = (0.229, 0.224, 0.225),
        pin_memory: bool = True,
        persistent_workers: bool = True,
        # Data augmentation parameters
        augmentation_enabled: bool = True,
        random_flip_p: float = 0.5,
        random_rotation_degrees: float = 15,
        color_jitter_brightness: float = 0.2,
        color_jitter_contrast: float = 0.2,
        color_jitter_saturation: float = 0.2,
        color_jitter_hue: float = 0.1,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.image_size = image_size
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        # Store augmentation parameters
        self.augmentation_enabled = augmentation_enabled
        self.random_flip_p = random_flip_p
        self.random_rotation_degrees = random_rotation_degrees
        self.color_jitter_brightness = color_jitter_brightness
        self.color_jitter_contrast = color_jitter_contrast
        self.color_jitter_saturation = color_jitter_saturation
        self.color_jitter_hue = color_jitter_hue
        self.save_hyperparameters()
        
        # Define transforms
        if self.augmentation_enabled:
            # Training transform with augmentation
            self.train_transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=self.random_flip_p),
                transforms.RandomRotation(degrees=self.random_rotation_degrees),
                transforms.ColorJitter(
                    brightness=self.color_jitter_brightness, 
                    contrast=self.color_jitter_contrast, 
                    saturation=self.color_jitter_saturation, 
                    hue=self.color_jitter_hue
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std)
            ])
        else:
            # Training transform without augmentation (same as val/test)
            self.train_transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std)
            ])
        
        # Validation and test transform (always without augmentation)
        self.val_test_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std)
        ])
        
        # Will be set during setup
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str = None):
        """Setup datasets for different stages"""
        
        if stage == "fit" or stage is None:
            # Load full training dataset
            full_train_dataset = Hotdog_NotHotdog(
                train=True, 
                transform=self.train_transform, 
                data_path=self.data_dir
            )
            
            # Split training data into train and validation
            total_size = len(full_train_dataset)
            val_size = int(self.val_split * total_size)
            train_size = total_size - val_size
            
            self.train_dataset, val_dataset_with_train_transform = random_split(
                full_train_dataset, 
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)  # For reproducibility
            )
            
            # Create validation dataset with validation transforms
            # We need to create a new dataset instance for validation with different transforms
            val_base_dataset = Hotdog_NotHotdog(
                train=True, 
                transform=self.val_test_transform, 
                data_path=self.data_dir
            )
            
            # Get the same indices as validation split
            val_indices = val_dataset_with_train_transform.indices
            self.val_dataset = torch.utils.data.Subset(val_base_dataset, val_indices)

        if stage == "test" or stage is None:
            # Load test dataset
            self.test_dataset = Hotdog_NotHotdog(
                train=False, 
                transform=self.val_test_transform, 
                data_path=self.data_dir
            )

    def train_dataloader(self):
        """Return training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False
        )

    def val_dataloader(self):
        """Return validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False
        )

    def test_dataloader(self):
        """Return test dataloader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False
        )

    def predict_dataloader(self):
        """Return prediction dataloader (same as test)"""
        return self.test_dataloader()

    def get_class_names(self):
        """Return class names"""
        # Create a temporary dataset to get class names
        temp_dataset = Hotdog_NotHotdog(
            train=True, 
            transform=self.val_test_transform, 
            data_path=self.data_dir
        )
        # Reverse the name_to_label mapping to get label_to_name
        label_to_name = {v: k for k, v in temp_dataset.name_to_label.items()}
        return [label_to_name[i] for i in range(len(label_to_name))]

    def get_dataset_info(self):
        """Return information about the dataset"""
        if self.train_dataset is None or self.val_dataset is None or self.test_dataset is None:
            self.setup()
        
        info = {
            "num_classes": 2,
            "class_names": self.get_class_names(),
            "train_size": len(self.train_dataset),
            "val_size": len(self.val_dataset),
            "test_size": len(self.test_dataset),
            "image_size": (3, self.image_size, self.image_size),
            "batch_size": self.batch_size
        }
        return info


if __name__ == "__main__":
    # Example usage and testing
    dm = HotdogDataModule(batch_size=16, val_split=0.2)
    dm.setup()
    
    print("Dataset Information:")
    info = dm.get_dataset_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\nDataloader Information:")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Test loading a batch
    print("\nTesting batch loading:")
    batch = next(iter(train_loader))
    images, labels = batch
    print(f"  Batch images shape: {images.shape}")
    print(f"  Batch labels shape: {labels.shape}")
    print(f"  Labels in batch: {labels.tolist()}")
