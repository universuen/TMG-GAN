import torch

import context

import src

if __name__ == '__main__':
    src.utils.set_random_state()
    gan = src.GAN(2)
    gan.fit()
    gan.test()
