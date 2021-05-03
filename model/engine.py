from tqdm import tqdm


def train_fn(train_dataloader, detector, optimizer, device, scheduler=None):
    detector.train()
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