import utils
from data_helper import train_data_set
from torch.utils.data import DataLoader

if __name__ == "__main__":

    log_file = 'lpg.txt'
    utils.writelog(file=log_file, log_info='begin to load data')
    train_set = train_data_set(
        image_dir='../data/train_data/',
        anno_path='../data/instances_train2017_small.json')
    train_set_load = DataLoader(train_set, batch_size=1, shuffle=False)
    utils.writelog(file=log_file, log_info='finished load data')

    for idx, data in enumerate(train_set_load):
        x, y = data[0], data[1]
