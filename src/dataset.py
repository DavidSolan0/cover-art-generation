import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms


class DatasetLoader:
    def __init__(self, root_path):
        self.root_path = root_path

    def create_dataset(self, image_size, batch_size, num_workers):
        transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        dataset = dset.ImageFolder(root=self.root_path, transform=transform)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        return dataloader
