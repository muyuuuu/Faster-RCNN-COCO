from torch._C import Size
import utils, os, torch, time, engine, model
from data_helper import train_data_set
from torch.utils.data import DataLoader
import torch.optim as optim

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda")

if __name__ == "__main__":

    log_file = 'log.txt'
    if os.path.exists(log_file):
        os.remove(log_file)
    utils.writelog(file=log_file,
                   log_info='=' * 10 + 'begin to load data' + '=' * 10)
    since = time.time()
    train_set = train_data_set(image_dir='../data/train_data/',
                               anno_path='../data/reconstruct-anno.json',
                               size=512)
    train_set_load = DataLoader(train_set, batch_size=4, shuffle=True)
    utils.writelog(file=log_file,
                   log_info='=' * 10 + 'finished load data' + '=' * 10 +
                   ',  ' + str(time.time() - since))

    # for images, targets in train_set_load:
    #     print(targets, '\n')
    #     break

    utils.writelog(file=log_file,
                   log_info='=' * 10 + 'begin to set model' + '=' * 10)
    since = time.time()
    start_epoch = 0

    detector = model.get_model()
    params = [p for p in detector.parameters() if p.requires_grad]
    optimizer = optim.Adam(params)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

    if os.path.exists('model.pth'):
        detector, start_epoch, optimizer, lr_scheduler = utils.get_checkpoint_state(
            "model.pth", detector, optimizer, lr_scheduler)
    detector.to(device)
    utils.writelog(file=log_file,
                   log_info='=' * 10 + 'finished to set model' + '=' * 10 +
                   ',   ' + str(time.time() - since))

    min_loss = 1000000000000
    for epoch in range(start_epoch, start_epoch + 5):
        loss_value = engine.train_fn(train_set_load, detector, optimizer,
                                     device, epoch, lr_scheduler)
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