"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""

import matplotlib.pyplot as plt
import re


def read(name):
    f = open(name, 'r')
    file = f.read()
    file = re.sub('\\[', '', file)
    file = re.sub('\\]', '', file)
    f.close()

    return [float(i) for idx, i in enumerate(file.split(','))]


def draw(mode):
    if mode == 'loss':
        train = read('./result/train_loss.txt')
        test = read('./result/test_loss.txt')
        plt.plot(train, 'r', label='train')
        plt.plot(test, 'b', label='validation')
        plt.legend(loc='lower left')


    elif mode == 'auc':
        train = read('./result/train_auc.txt')
        valid = read('./result/valid_auc.txt')
        plt.plot(train, 'r', label='train auc')
        plt.plot(valid, 'b', label='valid auc')
        plt.legend(loc='lower left')

    plt.xlabel('epoch')
    plt.ylabel(mode)
    plt.title('training result')
    plt.grid(True, which='both', axis='both')
    plt.show()


if __name__ == '__main__':
    draw(mode='loss')
    draw(mode='auc')
