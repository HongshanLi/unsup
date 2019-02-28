import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | mnist |imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--adv-epoches', type=int, default=25,
help='number of epochs to train for adversary')
parser.add_argument('--clf-epoches', type=int, default=25,
help='number of epochs to train for classfication')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam. default=0.999')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--outf', default='.', 
help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument("--n-classes", type=int, default=10, help="num of classes")
parser.add_argument("--debug", action="store_true", help="debug mode")

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    nc=3
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(root=opt.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Resize(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
    nc=3
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    nc=3

elif opt.dataset == 'mnist':
        dataset = dset.MNIST(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                           ]))
        nc=1

elif opt.dataset == 'fake':
    dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),
                            transform=transforms.ToTensor())
    nc=3

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output




class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # layer to detect real or false image
        self.adv_layers = nn.Sequential(
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

        # layer to classify digits
        self.classify_layers = nn.Sequential(
            # state size (ndf*8) x 4 x 4
            nn.Linear(ndf*8*4*4, opt.n_classes),
            nn.Softmax(dim=1)
        )



    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        validity = self.adv_layers(output).view(-1)

        return validity

    def classify(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)


        flat = output.view(output.shape[0], -1)
        class_label = self.classify_layers(flat)

        return class_label


def load_data():
    for i, data in enumerate(dataloader, 0):
        print(data[0].size())
        if i > 0:
            break

def adversarial_train(epoches):
    """
    Train generator and discriminator adversarially
    """
    netG = Generator(ngpu).to(device)
    netG.apply(weights_init)

    print(netG)

    netD = Discriminator(ngpu).to(device)
    netD.apply(weights_init)
    print(netD)


    criterion = nn.BCELoss()
    fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)

    real_label = 1
    fake_label = 0

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    if opt.debug == False:
        report_path = os.path.join(opt.outf, "training_report.txt")
        report = open(report_path, "w")
        report.write("Epoch, Batch, Loss_D, Loss_G, D(x), D(G(z))\n")

    for epoch in range(1, epoches+1):
        for i, data in enumerate(dataloader):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label, device=device)

            output = netD(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()


            report.write(
            '[%d/%d],[%d/%d],%.4f,%.4f,%.4f,%.4f / %.4f\n'
                  % (epoch, epoches , i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            if i % 100 == 0:
                vutils.save_image(real_cpu,
                        '%s/real_samples.png' % opt.outf,
                        normalize=True)
                fake = netG(fixed_noise)
                vutils.save_image(fake.detach(),
                        '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                        normalize=True)
                
                print(
                '[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                    % (epoch, epoches , i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            if opt.debug:
                if i > 0: 
                    break
            else:
                report.write(
                '[%d/%d],[%d/%d],%.4f,%.4f,%.4f,%.4f / %.4f\n'
                      % (epoch, epoches , i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
            
    report.close()
    return

def _one_hot_encode(labels, n_classes):
    """
    Args:
        labels: a tensor of shape (batch_size, 1)
    """
    # labels will be the index tensor
    batch_size = labels.size()[0]
    labels = labels.view(-1, 1)

    y_onehot = torch.FloatTensor(batch_size, n_classes).zero_()
    y_onehot.scatter_(1, labels, 1)

    return y_onehot


def classification_train(epoches, state_dict_path, report_name):
    """
    Train the discriminator to classfiy digits using
    its classify_layers
    Args:
        epoch: number of epochs to train for classify class labels
        state_dict: path to the trained discriminator parameters
    """

    netD = Discriminator(ngpu).to(device)

    if state_dict_path:
        print("Loading parameters from {}".format(state_dict_path))
        netD.load_state_dict(torch.load(state_dict_path))
    else:
        print("Training a fresh classifier")
        netD.apply(weights_init)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(netD.parameters(),
    lr=opt.lr, betas=(opt.beta1, opt.beta2))
    
    if opt.debug == False:
        f = open(report_name, "w")
        f.write("Epoch, Batch, Loss\n")
    for epoch in range(1, epoches+1):
        for i, (imgs, labels) in enumerate(dataloader):
            netD.zero_grad()
            imgs = imgs.to(device)
            batch_size = imgs.size(0)
            labels = labels.to(device)
            #labels = _one_hot_encode(labels, opt.n_classes)

            output = netD.classify(imgs)
            errD = criterion(output, labels)
            errD.backward()
            optimizer.step()
            
            if i % 50 == 0:
                print(
                '[%d/%d][%d/%d] Loss_D: %.4f'
                    % (epoch, epoches , i, len(dataloader),
                         errD.item()))
           
            if opt.debug:
                if i > 0:
                    break
            else:
                f.write(
                '[%d/%d],[%d/%d],%.4f\n'
                    % (epoch, epoches , i, len(dataloader),
                         errD.item()))


    f.close()
    return


def compare_efficiency(epoches):
    """
    Train a fresh discriminator for classfication for 10 epochs
    and then train a adverserially trained discriminator for
    10 epochs.
    See which one achieves higher classification accuracy faster
    """
    state_dict_path = "netD_epoch_{}.pth".format(opt.adv_epoches)
    state_dict_path = os.path.join(opt.outf, state_dict_path)

    report_path = os.path.join(opt.outf, "clf_report_adTrained.txt")
    classification_train(
        epoches=epoches,
        state_dict_path=state_dict_path,
        report_name=report_path
    )

    
    report_path = os.path.join(opt.outf, "clf_report_fresh.txt")
    classification_train(
        epoches=epoches,
        state_dict_path=None,
        report_name=report_path
    )

    return




if __name__=="__main__":
    #adversarial_train(epoches=opt.adv_epoches)
    compare_efficiency(epoches=opt.clf_epoches)
