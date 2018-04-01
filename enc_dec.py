import argparse
import numpy
import torch
from torch.autograd import Variable
import torch.optim as optim
from ind_rnn_net import EncDec
from torch.utils.data import DataLoader, Dataset


class OwnDataSet(Dataset):
    def __init__(self, batch_size, is_train=True):
        # 訓練データ
        t = numpy.linspace(0, 5*numpy.pi, 500)
        self.train = 10*numpy.sin(t).reshape(-1,1)
        self.train = numpy.tile(numpy.abs(self.train), (4*batch_size, 1)).astype('f')
        self.train = self.train.reshape(batch_size, -1, 1)

        # テストデータ
        t = numpy.linspace(0, 4*numpy.pi, 400)
        self.valid = 10*numpy.sin(t).reshape(-1,1)
        self.valid = numpy.concatenate((numpy.random.randn(100).reshape(100,1), self.valid), axis=0)
        self.valid = numpy.tile(numpy.abs(self.valid), (4, 1)).astype('f')
        self.valid = self.valid.reshape(1, -1, 1)

        self.is_train = is_train

    def __len__(self):
        if self.is_train:
            return len(self.train)
        else:
            return len(self.valid)

    def __getitem__(self, index):
        if self.is_train:
            return self.train[index].astype(numpy.float32)
        else:
            return self.valid[index].astype(numpy.float32)


def train(epoch, args, model, optimizer, data_loader):
    model.train()
    model.is_train = True
    for batch_idx, data in enumerate(data_loader):
        if torch.cuda.is_available():
            data = data.cuda()
        data = Variable(data)
        optimizer.zero_grad()
        loss = model(data)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.data[0]))


def test(args, model, optimizer, data_loader):
    model.eval()
    model.is_train = False
    test_loss = 0
    for data in data_loader:
        if torch.cuda.is_available():
            data = data.cuda()
        data = Variable(data, volatile=True)
        loss = model(data)
        test_loss += loss.data # sum up batch loss
        # test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = model.out # get the index of the max log-probability
        # print(model.out)

    print('\nTest set: Average loss: {:.4f}\n'.format(
        test_loss.numpy()[0], len(data_loader.dataset)))


def main(args):
    train_dataset = OwnDataSet(args.batchsize, True)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=4)
    test_dataset = OwnDataSet(args.batchsize, False)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=4)

    model = EncDec(args.inputsize, args.unit, args.slotsize, args.memorysize)

    if torch.cuda.is_available():
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(1, args.epochs + 1):
        train(epoch, args, model, optimizer, train_loader)
        test(args, model, optimizer, train_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chainer example: PRNN')
    parser.add_argument('--inputsize', '-in', type=int, default=1)
    parser.add_argument('--slotsize', '-sl', type=int, default=32)
    parser.add_argument('--memorysize', '-m', type=int, default=64)
    parser.add_argument('--batchsize', '-b', type=int, default=16,
                        help='number of sentence pairs in each mini-batch')
    parser.add_argument('--epochs', '-e', type=int, default=50,
                        help='number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--resume', '-r', default='',
                        help='resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=128,
                        help='number of units')
    parser.add_argument('--log-interval', type=int, default=200,
                        help='number of iteration to show log')
    parser.add_argument('--validation-interval', type=int, default=4000,
                        help='number of iteration to evlauate the model '
                        'with validation dataset')
    parser.add_argument('--out', '-o', default='result',
                        help='directory to output the result')
    args = parser.parse_args()

    main(args)
