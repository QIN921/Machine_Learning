from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.utils import to_categorical


def main():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    plt.imshow(test_images[0], cmap=plt.cm.binary)

    train_labels = to_categorical(train_labels)  # 独热码


if __name__ == '__main__':
    main()
    plt.show()
