from data_helper import valid_data_set
from torch.utils.data import DataLoader


def test():
    valid_set = valid_data_set(image_dir='../data/valid_data/')
    valid_set_load = DataLoader(valid_set, batch_size=1, shuffle=False)

    for data in valid_set_load:
        _, image_name = data
        print(image_name[0])


if __name__ == "__main__":
    test()