from tqdm import tqdm
import time
import utils


def train_fn(train_dataloader, detector, optimizer, device, epoch, scheduler):
    detector.train()
    loss_value = 0
    cnt = 0
    for images, target in tqdm(train_dataloader):
        cnt += 1
        images = list(image.to(device) for image in images)
        # it's key:value for t in targets.items
        # This is the format the fasterrcnn expects for targets
        targets = []
        for l, b in zip(target['labels'], target['boxes']):
            d = {}
            d['labels'] = l.view(-1).to(device)
            d['boxes'] = b.view(-1, 4).to(device)
            targets.append(d)
        loss_dict = detector(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        scheduler.step()

        if cnt % 1000 == 999:
            cnt = 0
            utils.save_checkpoint_state("tmp.pth", epoch, detector, optimizer,
                                        scheduler)

    return loss_value


def predict(val_dataloader, detector, device):
    results = []
    for images, image_names in tqdm(val_dataloader):
        images = list(image.to(device) for image in images)
        model_time = time.time()
        outputs = detector(images)
        model_time = time.time() - model_time
        for i, image in enumerate(images):
            boxes = (outputs[i]["boxes"].detach().cpu().numpy().tolist())
            scores = outputs[i]["scores"].detach().cpu().numpy()
            labels = outputs[i]["labels"].detach().cpu().numpy()
            image_id = image_names[i]
            for b, s, l in zip(boxes, scores, labels):
                if s > 0.1:
                    result = {  # Store the image id and boxes and scores in result dict.
                        "image_id": image_id,
                        "boxes": b,
                        "scores": s.astype(float),
                        "labels": l.astype(float),
                    }
                    results.append(result)
    return results