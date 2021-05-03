import utils, os, torch, time
from data_helper import train_data_set
from torch.utils.data import DataLoader
import torchvision.models.detection as td
import torch.optim as optim

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda")

if __name__ == "__main__":

    log_file = 'log.txt'
    utils.writelog(file=log_file,
                   log_info='=' * 10 + 'begin to load data' + '=' * 10)
    since = time.time()
    train_set = train_data_set(
        image_dir='../data/train_data/',
        anno_path='../data/instances_train2017_small.json')
    train_set_load = DataLoader(train_set, batch_size=8, shuffle=True)
    utils.writelog(file=log_file,
                   log_info='=' * 10 + 'finished load data' + '=' * 10 +
                   ',  ' + time.time() - since)

    utils.writelog(file=log_file,
                   log_info='=' * 10 + 'begin to set model' + '=' * 10)
    since = time.time()
    start_epoch = 0
    anchor_sizes = ((256, ), (512, ), (1024, ))
    aspect_ratios = ((0.5, 1.0, 2.0), ) * len(anchor_sizes)
    rpn_anchor_generator = td.anchor_utils.AnchorGenerator(
        anchor_sizes, aspect_ratios)
    # https://github.com/pytorch/vision/blob/master/torchvision/models/detection/faster_rcnn.py
    detector = td.__dict__['fasterrcnn_resnet50_fpn'](
        num_classes=515,
        rpn_anchor_generator=rpn_anchor_generator,
        pretrained=False)
    params = [p for p in detector.parameters() if p.requires_grad]
    optimizer = optim.Adam(params)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

    if os.path.exists('model.pth'):
        detector, start_epoch, optimizer, lr_scheduler = utils.get_checkpoint_state(
            "model.pth", detector, optimizer, lr_scheduler)
    detector.to(device)
    utils.writelog(file=log_file,
                   log_info='=' * 10 + 'finished to set model' + '=' * 10 +
                   ',   ' + time.time() - since)

    min_loss = 1000000000000
    for epoch in range(start_epoch, start_epoch + 5):
        loss_value = engine.train_fn(train_set_load, detector, optimizer,
                                     device, lr_scheduler)
        print("epoch = {}, Training_loss = {}".format(epoch, loss_value))
        utils.writelog(file=log_file,
                       log_info="epoch = {}, Training_loss = {}".format(
                           epoch, loss_value))
        # Set the threshold as per needs
        if loss_value < min_loss:
            min_loss = loss_value
            utils.save_checkpoint_state("model.pth", epoch, detector,
                                        optimizer, lr_scheduler)
            utils.writelog(
                file=log_file,
                log_info=">>>>>>>>>>>>epoch = {}, save model<<<<<<<<<<<".
                format(epoch))

    print("-" * 25)
    utils.writelog(file=log_file, log_info="-" * 25)
    utils.writelog(file=log_file, log_info="Model Trained and Saved to Disk")