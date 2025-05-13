import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
from torch.utils.data import random_split
import swanlab
import os
import time
import copy

if __name__ == '__main__':

    swanlab.init(
        project_name="flower_classification",  # Name of the project in SwanLab
        experiment_name="test_1",
    )

    # Set data directory
    data_dir = './flower_dataset'

    # Data augmentation and normalization for training and validation
    data_transforms = transforms.Compose([
            # GRADED FUNCTION: Add five data augmentation methods, Normalizating and Tranform to tensor
            ### START SOLUTION HERE ###
            # five data augmentation methods
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.RandomRotation(60),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomResizedCrop(224),
            # transform to tensor and normalization
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ### END SOLUTION HERE ###
    ])

    # Load the entire dataset
    full_dataset = datasets.ImageFolder(data_dir, data_transforms)

    # Automatically split into 80% train and 20% validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Use DataLoader for both train and validation datasets
    batch_size = 16
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    dataloaders = {'train': train_loader, 'val': val_loader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

    # Get class names from the dataset
    class_names = full_dataset.classes

    # Load pre-trained model and modify the last layer
    model = models.resnet18(pretrained=True)


    # GRADED FUNCTION: Modify the last fully connected layer of model
    ### START SOLUTION HERE ###
    # Modify the last fully connected layer of model
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    ### END SOLUTION HERE ###



    # GRADED FUNCTION: Define the loss function
    ### START SOLUTION HERE ###
    # Define the loss function
    criterion = nn.CrossEntropyLoss()
    ### END SOLUTION HERE ###

    # GRADED FUNCTION: Define the optimizer
    ### START SOLUTION HERE ###
    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    ### END SOLUTION HERE ###

    # Learning rate scheduler
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=0.00001)

    # Training function
    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Print and record learning rate
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Learning Rate: {current_lr:.6f}')
            swanlab.log({'LR':current_lr})  # record learning rate

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                
                # write loss and accuracy to Swanlab
                swanlab.log({f'{phase}/loss': epoch_loss, f'{phase}/acc': epoch_acc})

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    save_dir = './work_dir/exp6'
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(save_dir, f'epoch_{epoch}.pth'))

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}') 

        model.load_state_dict(best_model_wts)
        return model

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = train_model(model, criterion, optimizer, scheduler, num_epochs=25)