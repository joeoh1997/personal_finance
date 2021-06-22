from crypto_trader.image_labeler import create_dataset

data_path='E:/data/streams/XRPEUR/', #'E:/validation/'
play_path='data/streams/XRPEUR/random_play/'

create_classification_dataset = True


if create_classification_dataset:
    create_dataset(data_path)

