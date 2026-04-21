import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


# Following link has the meana nd std values
# https://pytorch.org/vision/stable/models/resnet.html

def get_dataloader(data_dir='./data', batch_size=32):
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = CIFAR10(root=data_dir, train=True, download=False, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader, dataset
