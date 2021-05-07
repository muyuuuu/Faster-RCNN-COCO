from tqdm import tqdm
import time
import utils


def train_fn(train_dataloader, detector, optimizer, device, epoch, scheduler):
    detector.train()
    loss_value = 0
    for i, images, targets in enumerate(tqdm(train_dataloader)):
        images = list(image.to(device) for image in images)
        # it's key:value for t in targets.items
        # This is the format the fasterrcnn expects for targets
        target = []
        for l, b in zip(targets['labels'], targets['boxes']):
            d = {}
            d['labels'] = l.view(-1).to(device)
            d['boxes'] = b.view(-1, 4).to(device)
            target.append(d)
        loss_dict = detector(images, target)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        scheduler.step()

        if i % 2000 == 1999:
            utils.save_checkpoint_state("model1.pth", epoch, detector,
                                        optimizer, scheduler)

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