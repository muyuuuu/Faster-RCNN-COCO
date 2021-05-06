from tqdm import tqdm
import time


def train_fn(train_dataloader, detector, optimizer, device, scheduler=None):
    detector.train()
    loss_value = 0
    for images, targets in tqdm(train_dataloader):
        images = list(image.to(device) for image in images)
        # it's key:value for t in targets.items
        # This is the format the fasterrcnn expects for targets
        targets = [{
            k: v.to(device)
            for k, v in t.items() if not isinstance(v, list)
        } for t in targets]
        loss_dict = detector(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

    return loss_value


def predict(val_dataloader, detector, device):
    results = []
    for images, image_names in tqdm(val_dataloader):
        images = list(image.to(device) for image in images)
        model_time = time.time()
        outputs = detector(images)
        model_time = time.time() - model_time
        for i, image in enumerate(images):
            boxes = (outputs[i]["boxes"].data.cpu().numpy())
            scores = outputs[i]["scores"].data.cpu().numpy()
            labels = outputs[i]["labels"].data.cpu().numpy()
            image_id = image_names[i]
            result = {  # Store the image id and boxes and scores in result dict.
                "image_id": image_id,
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
            }
            results.append(result)

    return results