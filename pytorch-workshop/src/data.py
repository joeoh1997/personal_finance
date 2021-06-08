from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SimCLRDataset(Dataset):
    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        chosen_image_np = self.x[item]

        chosen_image_pil = Image.fromarray(chosen_image_np)

        rrc = transforms.RandomResizedCrop(32)
        cj = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        tt = transforms.ToTensor()

        transformed_image1 = rrc(chosen_image_pil)
        transformed_image1 = cj(transformed_image1)
        transformed_image1_tensor = tt(transformed_image1)

        transformed_image2 = rrc(chosen_image_pil)
        transformed_image2 = cj(transformed_image2)
        transformed_image2_tensor = tt(transformed_image2)

        return transformed_image1_tensor, transformed_image2_tensor
