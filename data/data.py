import torch
import os
import shutil
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

with open('dataset/tiny-imagenet/tiny-imagenet-200/val/val_annotations.txt') as f:
    for line in f:
        fn, cls, *_ = line.split('\t')
        os.makedirs(f'dataset/tiny-imagenet/tiny-imagenet-200/val/{cls}', exist_ok=True)

        shutil.copyfile(f'dataset/tiny-imagenet/tiny-imagenet-200/val/images/{fn}', f'dataset/tiny-imagenet/tiny-imagenet-200/val/{cls}/{fn}')

shutil.rmtree('dataset/tiny-imagenet/tiny-imagenet-200/val/images')


transform = T.Compose([
    T.Resize((224, 224)),  
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


tiny_imagenet_dataset_train = ImageFolder(root='dataset/tiny-imagenet/tiny-imagenet-200/train', transform=transform)
tiny_imagenet_dataset_val = ImageFolder(root='dataset/tiny-imagenet/tiny-imagenet-200/val', transform=transform)

train_loader = torch.utils.data.DataLoader(tiny_imagenet_dataset_train, batch_size=32, shuffle=True, num_workers=8)
val_loader = torch.utils.data.DataLoader(tiny_imagenet_dataset_val, batch_size=32, shuffle=False)

print(f"Length of train dataset: {len(tiny_imagenet_dataset_train)}")
print(f"Length of val dataset: {len(tiny_imagenet_dataset_val)}")