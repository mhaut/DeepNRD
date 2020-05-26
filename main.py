
import argparse
import auxil
from DA_HSI import ROhsi
from hyper_pytorch import *
import models
import torch
from torchvision.transforms import *


def load_hyper(args):
    data, labels, numclass = auxil.loadData(args.dataset, num_components=args.components, preprocess=args.preprocessing)
    pixels, labels = auxil.createImageCubes(data, labels, windowSize=args.spatialsize, removeZeroLabels = True)
    bands = pixels.shape[-1]; numberofclass = len(np.unique(labels))
    x_train, x_test, y_train, y_test = auxil.split_data(pixels, labels, args.tr_percent)
    if args.use_val: x_val, x_test, y_val, y_test = auxil.split_data(x_test, y_test, args.val_percent)
    del pixels, labels
    if args.p != 0: transform_train = ROhsi(probability = args.p, sh = args.sh, r1 = args.r1,)
    else: transform_train = None
    train_hyper = HyperData((np.transpose(x_train, (0, 3, 1, 2)).astype("float32"),y_train), transform_train)
    test_hyper  = HyperData((np.transpose(x_test, (0, 3, 1, 2)).astype("float32"),y_test), None)
    if args.use_val: val_hyper   = HyperData((np.transpose(x_val, (0, 3, 1, 2)).astype("float32"),y_val), None)
    kwargs = {'num_workers': 0, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_hyper, batch_size=args.tr_bsize, shuffle=True, **kwargs)
    test_loader  = torch.utils.data.DataLoader(test_hyper, batch_size=args.te_bsize, shuffle=False, **kwargs)
    if args.use_val: val_loader  = torch.utils.data.DataLoader(val_hyper, batch_size=args.val_bsize, shuffle=False, drop_last=True, **kwargs)
    else: val_loader = None
    return train_loader, val_loader, test_loader, numberofclass, bands
   

def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    model.train()
    total_loss = 0.0
    predictions = []; real = []
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item()
        pred = outputs.data.max(1)[1]
        predictions += list(pred.data.cpu().numpy())
        real += list(targets.data.cpu().numpy())
        loss.backward()
        optimizer.step()
    accs = auxil.getaccuracy(real, predictions)
    return (total_loss / len(trainloader), accs)


def test(testloader, model, criterion, epoch, use_cuda):
    model.eval()
    total_loss = 0.0
    predictions = []; real = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            pred = outputs.data.max(1)[1]
            predictions += list(pred.data.cpu().numpy())
            real += list(targets.data.cpu().numpy())
    accs = auxil.getaccuracy(real, predictions)
    return (total_loss / len(testloader), accs)


def predict(testloader, model, criterion, use_cuda):
    model.eval()
    predicted = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda: inputs = inputs.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            [predicted.append(a) for a in model(inputs).data.cpu().numpy()] 
    return np.array(predicted)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='IP', type=str, help='Dataset')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', default=500, type=int, help='Number of total epochs')
    parser.add_argument('--components', default=1, type=int, help='Dimensionality reduction')
    parser.add_argument('--spatialsize', default=23, type=int, help='Patch size')
    parser.add_argument('--tr_percent', default=0.10, type=float, metavar='N', 
                        help='Train set size')
    parser.add_argument('--val_percent', default=0.1, type=float, metavar='N', 
                        help='Train set size')
    parser.add_argument('--tr_bsize', default=100, type=int, metavar='N',
                        help='Train batch size')
    parser.add_argument('--val_bsize', default=2000, type=int, metavar='N',
                        help='Test batch size')
    parser.add_argument('--te_bsize', default=2000, type=int, metavar='N',
                        help='Test batch size')
    parser.add_argument("--verbose", action='store_true', help="Verbose? Default NO")
    parser.add_argument("--use_val", action='store_true', help="Validation? Default NO")
    parser.add_argument('--schedule', type=int, nargs='+', default=[350],
                            help='Decrease learning rate at these epochs.')
    parser.add_argument('--preprocessing', default='standard', type=str, help='Dataset')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, help='weight decay (default: 1e-4)')

    # DropBlock
    parser.add_argument('--dropprob', default=0, type=float, help='DropBlock probability')
    parser.add_argument('--blocksize', default=3, type=int, help='size of block drop')

    # Dropout
    parser.add_argument('--simple_dropout', default=0, type=float, help='Normal dropout probability')

    # Random Erasing
    parser.add_argument('--p', default=0, type=float, help='Random Erasing probability')
    parser.add_argument('--sh', default=0.3, type=float, help='max erasing area')
    parser.add_argument('--r1', default=0.2, type=float, help='aspect of erasing area')


    args = parser.parse_args()
    state = {k: v for k, v in args._get_kwargs()}

    trainloader, valloader, testloader, num_classes, bands = load_hyper(args)

    # Use CUDA
    use_cuda = torch.cuda.is_available()
    if use_cuda: torch.backends.cudnn.benchmark = True

    model = models.SimpleCNN(bands, args.spatialsize, num_classes, args.simple_dropout, args.dropprob, args.blocksize)
    if use_cuda: model = model.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=0.1)


    best_acc = -1
    for epoch in range(args.epochs):
        train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
        if args.use_val: val_loss, val_acc = test(valloader, model, criterion, epoch, use_cuda)
        else: val_loss, val_acc = test(testloader, model, criterion, epoch, use_cuda)

        if args.verbose: print("EPOCH ["+str(epoch)+"/"+str(args.epochs)+"] TRAIN LOSS", train_loss, \
                                "TRAIN ACCURACY", train_acc, \
                                "LOSS", val_loss, "ACCURACY", val_acc, best_acc)
        # save model
        if val_acc > best_acc:
            state = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'acc': val_acc,
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),
            }
            torch.save(state, "best_model"+str(args.p)+".pth.tar")
            best_acc = val_acc
        scheduler.step()
    checkpoint = torch.load("best_model"+str(args.p)+".pth.tar")
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda)
    if args.verbose: print("FINAL:      LOSS", test_loss, "ACCURACY", test_acc)
    classification, confusion, results = auxil.reports(np.argmax(predict(testloader, model, criterion, use_cuda), axis=1), np.array(testloader.dataset.__labels__()))
    print(args.dataset, results)


if __name__ == '__main__':
    main()
