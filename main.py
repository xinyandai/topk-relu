import tqdm
import argparse
import torch.optim as optim

from models import *
from dataloaders import *
from logger import Logger


NETWORK         = None
DATASET_LOADER  = None
LOGGER          = None
LOSS_FUNC       = nn.CrossEntropyLoss()


network_choices = {
    'resnet18'  : ResNet18,
    'resnet34'  : ResNet34,
    'resnet50'  : ResNet50,
    'resnet101' : ResNet101,
    'resnet152' : ResNet152,
    'vgg11'     : vgg11,
    'vgg13'     : vgg13,
    'vgg16'     : vgg16,
    'vgg19'     : vgg19,
    'dense'     : densenet_cifar,
    'fcn'       : FCN
}

data_loaders = {
    'mnist':    minst,
    'cifar10':  cifar10,
    'cifar100': cifar100,
    'stl10':    stl10,
    'svhn':     svhn,
    'tinyimg':  tinyimgnet
}

classes_choices = {
    'mnist':    10,
    'cifar10':  10,
    'cifar100': 100,
    'stl10':    10,
    'svhn':     10,
    'tinyimg':  200
}


def get_config(args):
    global LOGGER
    global NETWORK
    global DATASET_LOADER

    NETWORK = network_choices[args.network]
    DATASET_LOADER = data_loaders[args.dataset]
    args.num_classes = classes_choices[args.dataset]
    LOGGER = Logger(args.logdir)

    args.no_cuda = args.no_cuda or not torch.cuda.is_available()


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='Gradient Quantization Samples')
    parser.add_argument('--network', type=str, default='resnet18', choices=network_choices.keys())
    parser.add_argument('--dataset', type=str, default='cifar10', choices=data_loaders.keys())

    parser.add_argument('--logdir', type=str, required=True,
                        help='For Saving the logs')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=350, metavar='N',
                        help='number of epochs to train (default: 350)')

    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='M',
                        help='weight decay momentum (default: 5e-4)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    args = parser.parse_args()
    get_config(args)

    torch.manual_seed(args.seed)
    device = torch.device("cpu" if args.no_cuda else "cuda")

    train_loader, test_loader = DATASET_LOADER(args)
    model = NETWORK(num_classes=args.num_classes).to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=args.momentum, weight_decay=args.weight_decay)

    if args.dataset == 'mnist':
        epochs = []
        lrs = []
        args.epochs = 20
    else:
        epochs = [51, 71]
        lrs = [0.01, 0.005]
        args.epochs = 150

    for epoch in range(1, args.epochs + 2):
        for i_epoch, i_lr in zip(epochs, lrs):
            if epoch == i_epoch:
                optimizer = optim.SGD(model.parameters(), lr=i_lr,
                                      momentum=args.momentum, weight_decay=5e-4)

        origin_train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader, epoch)


def test(args, model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += LOSS_FUNC(output, target).sum().item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest Epoch {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        epoch, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / len(test_loader.dataset)


def origin_train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(tqdm.tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = LOSS_FUNC(output, target)
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    main()
