import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as T


def get_transforms():
    return T.Compose([
        T.Resize((224, 224)),  
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_dataloaders(base_path='dataset/tiny-imagenet-200', batch_size=32, num_workers=2):
    transform = get_transforms()
    
    # Load datasets
    train_dataset = ImageFolder(root=f'{base_path}/train', transform=transform)
    val_dataset = ImageFolder(root=f'{base_path}/val', transform=transform)
    
    print(f"Length of train dataset: {len(train_dataset)}")
    print(f"Length of val dataset: {len(val_dataset)}")
    
    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader