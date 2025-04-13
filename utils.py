import torch
import matplotlib.pyplot as plt


def show_train_hist(hist, path='train.png'):
    plt.figure()
    plt.plot(hist['D_losses'], label='D_loss')
    plt.plot(hist['G_losses'], label='G_loss')
    plt.legend()
    plt.savefig(path)
    plt.close()

def normal_init(m, mean, std):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()