from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
import torch
import os


dt_resize = transforms.Resize((32,32))
dt_resize2 = transforms.Resize((64,64))
dt_gray3ch = transforms.Grayscale(num_output_channels=3)
dt_tensor = transforms.ToTensor()
dt_imagenet = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

data_transforms_imagenet = transforms.Compose(
    [
        #dt_resize2,
        dt_gray3ch, #transforms.Grayscale(num_output_channels=3),
        dt_tensor, #transforms.ToTensor(),
        # Normalize input channels using mean values and standard deviations of ImageNet.
        dt_imagenet#transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

data_transforms_3channel = transforms.Compose(
    [
        #dt_resize2,
        dt_gray3ch, #transforms.Grayscale(num_output_channels=3),
        dt_tensor# transforms.ToTensor()
    ]
)

data_transforms_imagenet32 = transforms.Compose(
    [
        dt_resize,
        dt_gray3ch, #transforms.Grayscale(num_output_channels=3),
        dt_tensor, #transforms.ToTensor(),
        # Normalize input channels using mean values and standard deviations of ImageNet.
        dt_imagenet#transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

data_transforms_3channel32 = transforms.Compose(
    [
        dt_resize,
        dt_gray3ch, #transforms.Grayscale(num_output_channels=3),
        dt_tensor# transforms.ToTensor()
    ]
)


def load_kfold_dataset(DATA_DIR, imagenet=True):
    print('--> Loading data from ', DATA_DIR)
    if imagenet:
        data_transforms = data_transforms_imagenet
    else:
        data_transforms = data_transforms_3channel

    # create dataloaders
    train_image_datasets = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), data_transforms)
    val_image_datasets = datasets.ImageFolder(os.path.join(DATA_DIR, "validation"), data_transforms)
    dataset = ConcatDataset([train_image_datasets, val_image_datasets])

    return dataset


def load_trainval_dataset(DATA_DIR, batch_size=4, imagenet=True):
    print('--> Loading data from ',DATA_DIR)
    if imagenet:
        data_transforms = data_transforms_imagenet
    else:
        data_transforms = data_transforms_3channel

    # create dataloaders
    train_image_datasets = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), data_transforms)
    train_loader = torch.utils.data.DataLoader(dataset=train_image_datasets, shuffle=True, batch_size=batch_size)

    val_image_datasets = datasets.ImageFolder(os.path.join(DATA_DIR, "validation"), data_transforms)
    val_loader = torch.utils.data.DataLoader(dataset=val_image_datasets, shuffle=False, batch_size=batch_size)

    # merge train & val data loaders
    dataloaders = {'train': train_loader, 'validation': val_loader}
    dataset_sizes = {'train': len(train_image_datasets), 'validation': len(val_image_datasets)}

    return dataloaders, dataset_sizes


def load_test_dataset(DATA_DIR, batch_size=1, imagenet=True):
    print('--> Loading data from ', DATA_DIR)
    if imagenet:
        data_transforms = data_transforms_imagenet
    else:
        data_transforms = data_transforms_3channel

    test_image_datasets = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), data_transforms)
    test_loader = torch.utils.data.DataLoader(dataset=test_image_datasets, shuffle=False, batch_size=batch_size)

    dataloaders = {'test': test_loader}
    dataset_sizes = {'test': len(test_image_datasets)}

    return dataloaders, dataset_sizes


def load_test_dataset_to32x32(DATA_DIR, batch_size=1, imagenet=True):
    print('--> Loading data from ', DATA_DIR)
    if imagenet:
        data_transforms = data_transforms_imagenet32
    else:
        data_transforms = data_transforms_3channel32

    test_image_datasets = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), data_transforms)
    test_loader = torch.utils.data.DataLoader(dataset=test_image_datasets, shuffle=False, batch_size=batch_size)

    dataloaders = {'test': test_loader}
    dataset_sizes = {'test': len(test_image_datasets)}

    return dataloaders, dataset_sizes