from torch.utils.data import Dataset
from PIL import Image
import glob


# torch版本 2.0.0+cu118,python==3.8

class Pix2PixDataset(Dataset):
    def __init__(self, imgs_path, annos_path, transform=None):
        self.imgs = sorted(glob.glob(imgs_path + "/*.png"))
        self.annos = sorted(glob.glob(annos_path + "/*.png"))
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert("RGB")
        anno = Image.open(self.annos[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
            anno = self.transform(anno)
        return img, anno
