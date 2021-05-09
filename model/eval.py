import engine, json, torch, os, model, utils
from data_helper import valid_data_set
from torch.utils.data import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda")


def test():
    valid_set = valid_data_set(image_dir='/home/liyanni/1307/ljw/val_data/',
                               size=512)
    valid_set_load = DataLoader(valid_set, batch_size=1, shuffle=False)

    detector = model.get_model(516)
    checkpoint = torch.load('model.pth')
    detector.load_state_dict(checkpoint['model_state_dict'])
    detector.to(device)
    detector.eval()
    result = engine.predict(val_dataloader=valid_set_load,
                            detector=detector,
                            device=device)
    with open('result.json', 'w') as f:
        json.dump(result, f)


if __name__ == "__main__":
    test()