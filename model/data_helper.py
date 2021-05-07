import os, torch, json
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class train_data_set(Dataset):
    def __init__(self, image_dir, anno_path, size):
        super().__init__()
        self.image_dir = image_dir
        self.size = size
        self.transform = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize((0.44, 0.38, 0.30), (0.012, 0.013, 0.012))
        ])

        self.data = []
        with open(anno_path, 'r') as f:
            self.data = json.load(f)

    # 数据集很大时，要在 getitem 中获取
    def __getitem__(self, idx):
        # 某个图片的信息
        img_info = self.data[idx]
        target = {}
        target['labels'] = torch.tensor(img_info['labels'], dtype=torch.int64)
        box = img_info['boxes']
        # print(box)
        boxes = []
        # 盒子坐标映射
        height_scale = self.size / img_info['height']
        width_scale = self.size / img_info['width']

        for i, item in enumerate(box):
            if i == 0 or i == 2:
                boxes.append(item * height_scale)
            elif i == 1 or i == 3:
                boxes.append(item * width_scale)
        # print(boxes)
        target['boxes'] = torch.tensor(boxes, dtype=torch.float64)
        path = self.image_dir + img_info['file_name']
        return self.transform(path), target

    def __len__(self):
        return len(self.data)


class valid_data_set(Dataset):
    def __init__(self, image_dir, size) -> None:
        super().__init__()
        self.size = size
        self.transform = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        # 获取文件路径下所有的图片名称
        self.root = image_dir
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_name = self.images[index]
        return self.transform(self.root + image_name), image_name