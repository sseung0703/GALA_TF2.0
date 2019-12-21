def Dataloader(name, home_path):
    if name == 'mnist':
        return MNIST(home_path)

def MNIST(home_path):
    from tensorflow.keras.datasets.mnist import load_data
    (train_images, train_labels), (val_images, val_labels) = load_data()
    
    return train_images, train_labels, val_images, val_labels

