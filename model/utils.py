import torch


def writelog(file, log_info):
    '''
    write log informantion to file
    '''
    with open(file, 'a') as f:
        f.write(log_info + '\n')


def get_checkpoint_state(model_path, model, optimizer, scheduler):
    # 恢复上次的训练状态
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return model, epoch, optimizer, scheduler


def save_checkpoint_state(path, epoch, model, optimizer, scheduler):

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }

    torch.save(checkpoint, "model.pth")
