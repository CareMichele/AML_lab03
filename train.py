import torch
from torch import nn
from models.custom_net import CustomNet
from data.data import get_dataloaders
from eval import validate
from utils.data_preparation import prepare_tiny_imagenet_val


def train(epoch, model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')

    return train_accuracy


if __name__ == "__main__":
    # Prepare validation set (reorganize folder structure)
    print("Preparing dataset...")
    prepare_tiny_imagenet_val('dataset/tiny-imagenet-200')
    
    # Load datasets and create dataloaders
    print("\nLoading Tiny-ImageNet datasets...")
    train_loader, val_loader = get_dataloaders(
        base_path='dataset/tiny-imagenet-200',
        batch_size=32,
        num_workers=2  
    )
    
    # Setup model
    print("\nInitializing model...")
    model = CustomNet().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # Training loop
    best_acc = 0
    num_epochs = 10
    
    print(f"\nStarting training for {num_epochs} epochs...\n")
    for epoch in range(1, num_epochs + 1):
        train(epoch, model, train_loader, criterion, optimizer)
        val_accuracy = validate(model, val_loader, criterion)
        best_acc = max(best_acc, val_accuracy)
        print()
    
    print(f'Best validation accuracy: {best_acc:.2f}%')