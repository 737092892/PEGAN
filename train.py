from torch.autograd import Variable
import time
import os
import torch
from config import *
from data_loader import load_dataset
from models.model import Generator
from models.model import Discriminator
from utils import show_train_hist, normal_init


def main():
    train_loader = load_dataset(DATA_PATH)  # 需确保data_loader.py已适配BATCH_SIZE

    G = Generator()
    D = Discriminator()
    G.weight_init(mean=INIT_MEAN, std=INIT_STD)
    D.weight_init(mean=INIT_MEAN, std=INIT_STD)
    G.cuda()
    D.cuda()

    BCE_loss = torch.nn.BCELoss()
    G_optimizer = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    D_optimizer = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

    train_hist = {
        'D_losses': [],
        'G_losses': [],
        'per_epoch_ptimes': [],
        'total_ptime': []
    }

    onehot = torch.zeros(2, 2).scatter_(1, torch.LongTensor([0, 1]).view(2, 1), 1).view(2, 2, 1, 1)
    fill1 = torch.zeros([2, 2, 30, 30])
    fill2 = torch.zeros([2, 2, 116, 116])
    for i in range(2):
        fill1[i, i] = fill2[i, i] = 1

    print('Training start!')
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        D_losses = []
        G_losses = []

        epoch_start_time = time.time()
        y_real_ = torch.ones(batch_size)
        y_fake_ = torch.zeros(batch_size)
        y_real_, y_fake_ = Variable(y_real_.cuda()), Variable(y_fake_.cuda())

        for x_, y_ in train_loader:
            # print(x_.shape) #torch.Size([32, 116, 116])
            # print(y_.shape) #torch.Size([32])

            # train discriminator D
            D.zero_grad()
            mini_batch = x_.size()[0]
            x_ = x_.unsqueeze(1)
            # print(x_.shape) #torch.Size([32, 1, 116, 116])
            if mini_batch != batch_size:
                y_real_ = torch.ones(mini_batch)
                y_fake_ = torch.zeros(mini_batch)
                y_real_, y_fake_ = Variable(y_real_.cuda()), Variable(y_fake_.cuda())

            # y_label_ = torch.zeros(mini_batch, 2)
            # y_label_.scatter_(1, y_.view(mini_batch, 1), 1)
            y_fill_ = fill2[y_]
            # x_, y_label_, y_real_, y_fake_ = Variable(x_.cuda()), Variable(y_label_.cuda()), Variable(y_real_.cuda()), Variable(y_fake_.cuda())
            x_, y_fill_ = Variable(x_.cuda()), Variable(y_fill_.cuda())

            D_result = D(x_, y_fill_, epoch, full=True).squeeze()

            D_real_loss = BCE_loss(D_result, y_real_)

            # fake data
            z_ = torch.rand((mini_batch, 100)).view(-1, 100, 1, 1)
            # print(z_.shape) #torch.Size([32, 100, 1, 1])
            y_ = (torch.rand(mini_batch, 1) * 2).type(torch.LongTensor).squeeze()
            y_label_ = onehot[y_]
            y_fill_ = fill2[y_]  # torch.Size([32, 2, 116, 116])
            if (epoch < 500):
                y_fill_ = fill1[y_]  # torch.Size([32, 2, 30, 30])

            z_, y_label_, y_fill_ = Variable(z_.cuda()), Variable(y_label_.cuda()), Variable(
                y_fill_.cuda())  # [32,100,1,1]


            G_result = G(z_, y_label_,
                         epoch)  # torch.Size([32, 100, 1, 1])+torch.Size([32, 2, 1, 1])--->torch.Size([32, 64, 30, 30])
            D_result = D(G_result, y_fill_, epoch, False).squeeze()

            D_fake_loss = BCE_loss(D_result, y_fake_)

            D_train_loss = D_real_loss + D_fake_loss

            D_train_loss.backward()
            D_optimizer.step()

            D_losses.append(D_train_loss.data.item())

            # train generator G
            G.zero_grad()

            z_ = torch.rand((mini_batch, 100)).view(-1, 100, 1, 1)

            y_ = (torch.rand(mini_batch, 1) * 2).type(torch.LongTensor).squeeze()
            y_label_ = onehot[y_]
            y_fill_ = fill2[y_]
            if (epoch < 500):
                y_fill_ = fill1[y_]
            z_, y_label_, y_fill_ = Variable(z_.cuda()), Variable(y_label_.cuda()), Variable(y_fill_.cuda())
            # y_label_ = torch.zeros(mini_batch, 10)
            # y_label_.scatter_(1, y_.view(mini_batch, 1), 1)

            # z_, y_label_ = Variable(z_.cuda()), Variable(y_label_.cuda())

            G_result = G(z_, y_label_, epoch)
            D_result = D(G_result, y_fill_, epoch, False).squeeze()
            G_result_t = G_result.permute(0, 1, 3, 2)
            # criterion = nn.L1Loss()

            # G_train_loss = BCE_loss(D_result, y_real_)
            G_train_loss = BCE_loss(D_result, y_real_) + 0.1 * torch.nn.MSELoss()(G_result, G_result_t)

            G_train_loss.backward()
            G_optimizer.step()

            G_losses.append(G_train_loss.data.item())
            print('G_train_loss:', G_train_loss.item(), '-------' 'D_train_loss:', D_train_loss.data.item())

            if (epoch % 49) == 0:
                # Save the checkpoints.
                torch.save(G.state_dict(), os.path.join(keep_dir, 'G_{}.pth'.format(epoch)))
                # torch.save(D.state_dict(), os.path.join(keep_dir, 'D_{}.pth'.format(epoch)))

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time

        print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % (
        (epoch + 1), num_epochs, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
        torch.mean(torch.FloatTensor(G_losses))))
        # fixed_p = root + 'Fixed_results/' + model + str(epoch + 1) + '.png'
        # show_result((epoch + 1), save=True, path=fixed_p)
        train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
        train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

    end_time = time.time()
    total_ptime = end_time - start_time
    train_hist['total_ptime'].append(total_ptime)

    print("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (
    torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), num_epochs, total_ptime))
    print("Training finish!... save training results")
    # torch.save(G.state_dict(), "MNIST_cGAN_results/generator_param.pkl")
    # torch.save(D.state_dict(), "MNIST_cGAN_results/discriminator_param.pkl")

    # with open('./Results/train.pkl', 'wb') as f:
    # pickle.dump(train_hist, f)

    show_train_hist(train_hist, save=True, path=r'')


if __name__ == "__main__":
    main()