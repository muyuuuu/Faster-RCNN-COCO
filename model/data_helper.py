from PIL import Image
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from torchvision.transforms import transforms


class train_data_set(Dataset):
    def __init__(self, image_dir, anno_path):
        super().__init__()
        self.image_dir = image_dir
        # COCO api class that loads COCO annotation file and prepare data structures
        self.coco = COCO(anno_path)
        # 获取 image 的 id，字典转为 list
        self.ids = list(self.coco.imgs.keys())
        self.transform = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    # 数据集很大时，要在 getitem 中获取
    def __getitem__(self, idx):
        # 某个图片的 id
        img_id = self.ids[idx]
        # 获取图片的 annotations 的 id
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        # 根据 annotations 的 id 获取 annotions
        target = self.coco.loadAnns(ann_ids)
        path = self.image_dir + self.coco.loadImgs(img_id)[0]['file_name']

        for d in target:
            # 进行判断，第一次加载修改了 key，第二次
            if 'bbox' in d.keys():
                v = {}
                v['boxes'] = d.pop('bbox')
                d['boxes'] = v['boxes']
                x = []
                for y in v['boxes']:
                    x.append(y)
                x[2], x[3] = x[0] + x[2], x[1] + x[3]
                d['boxes'] = torch.tensor(x, dtype=torch.float32)
                v = {}
                v['labels'] = d.pop('category_id')
                d['labels'] = torch.tensor(v['labels'] - 1, dtype=torch.int64)
        return self.transform(path), target

    def __len__(self):
        return len(self.ids)
