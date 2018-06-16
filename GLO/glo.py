import util
import laploss
import generator

import plac
from tqdm import tqdm

import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import SGD
from torchvision.datasets import LSUN
from torchvision import transforms
from torchvision.utils import make_grid
import time

def main(
        lsun_data_dir: ('Base directory for the LSUN data'),
        image_output_prefix: ('Prefix for image output',
                              'option', 'o')='glo',
        code_dim: ('Dimensionality of latent representation space',
                   'option', 'd', int)=128,
        epochs: ('Number of epochs to train',
                 'option', 'e', int)=25,
        use_cuda: ('Use GPU?',
                   'flag', 'gpu')=False,
        batch_size: ('Batch size',
                     'option', 'b', int)=128,
        lr_g: ('Learning rate for generator',
               'option', None, float)=1.,
        lr_z: ('Learning rate for representation_space',
               'option', None, float)=10.,
        max_num_samples: ('Cap on the number of samples from the LSUN dataset',
                          'option', 'n', int)=-1,
        init: ('Initialization strategy for latent represetation vectors',
               'option', 'i', str, ['pca', 'random'])='pca',
        n_pca: ('Number of samples to take for PCA',
                'option', None, int)=(64 * 64 * 3 * 2),
        loss: ('Loss type (Laplacian loss as in the paper, or L2 loss)',
               'option', 'l', str, ['lap_l1', 'l2'])='lap_l1',
):
    a = time.time()
    train_set = util.IndexedDataset(
        LSUN(lsun_data_dir, classes=['bedroom_train'], 
             transform=transforms.Compose([

                 transforms.Resize(64),
                 transforms.CenterCrop(64),
                 transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
             ]))
    )
    b = time.time()
    print("===train set===\n")
    print(b-a)

    a = time.time()
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, 
        shuffle=True, drop_last=True,
        num_workers=8, pin_memory=use_cuda,
    )
    b = time.time()
    print("===train loader===\n")
    print(b - a)
    # we don't really have a validation set here, but for visualization let us 
    # just take the first couple images from the dataset
    val_loader = torch.utils.data.DataLoader(train_set, shuffle=False, batch_size=8*8)

    if max_num_samples > 0:
        train_set.base.length = max_num_samples
        train_set.base.indices = [max_num_samples]

    # initialize representation space:
    if init == 'pca':
        from sklearn.decomposition import PCA
        a = time.time()
        # first, take a subset of train set to fit the PCA
        X_pca = np.vstack([
            X.cpu().numpy().reshape(len(X), -1)
            for i, (X, _, _)
             in zip(tqdm(range(n_pca // train_loader.batch_size), 'collect data for PCA'), 
                    train_loader)
        ])
        b = time.time()
        print("===pca loader===\n")
        print(b - a)
        print("perform PCA...")
        pca = PCA(n_components=code_dim)
        pca.fit(X_pca)
        # then, initialize latent vectors to the pca projections of the complete dataset
        a = time.time()
        Z = np.empty((len(train_loader.dataset), code_dim))
        for X, _, idx in tqdm(train_loader, 'pca projection'):
            Z[idx] = pca.transform(X.cpu().numpy().reshape(len(X), -1))
        b = time.time()
        print("===pca projection===\n")
        print(b - a)
    elif init == 'random':
        Z = np.random.randn(len(train_set), code_dim)

    Z = util.project_l2_ball(Z)

    g = util.maybe_cuda(generator.Generator(code_dim), use_cuda)
    loss_fn = laploss.LapLoss(max_levels=3) if loss == 'lap_l1' else nn.MSELoss()
    zi = util.maybe_cuda(torch.zeros((batch_size, code_dim)),use_cuda)
    zi = Variable(zi, requires_grad=True)
    optimizer = SGD([
        {'params': g.parameters(), 'lr': lr_g}, 
        {'params': zi, 'lr': lr_z}
    ])

    Xi_val, _, idx_val = next(iter(val_loader))
    util.imsave('target.png',
           make_grid(Xi_val.cpu() / 2. + 0.5, nrow=8).numpy().transpose(1, 2, 0))

    for epoch in range(epochs):
        losses = []
        progress = tqdm(total=len(train_loader), desc='epoch % 3d' % epoch)

        for i, (Xi, yi, idx) in enumerate(train_loader):
            a = time.time()
            Xi = Variable(util.maybe_cuda(Xi, use_cuda))
            zi.data = util.maybe_cuda(torch.FloatTensor(Z[idx.numpy()]), use_cuda)

            optimizer.zero_grad()
            rec = g(zi)
            loss = loss_fn(rec, Xi)
            loss.backward()
            optimizer.step()

            Z[idx.numpy()] = util.project_l2_ball(zi.data.cpu().numpy())

            losses.append(loss.data[0])
            progress.set_postfix({'loss': np.mean(losses[-100:])})
            progress.update()
            b = time.time()
            print("===1 data===\n")
            print(b - a)
        progress.close()

        # visualize reconstructions
        rec = g(Variable(util.maybe_cuda(torch.FloatTensor(Z[idx_val.numpy()]), use_cuda)))
        util.imsave('%s_rec_epoch_%03d.png' % (image_output_prefix, epoch),
               make_grid(rec.data.cpu() / 2. + 0.5, nrow=8).numpy().transpose(1, 2, 0))


if __name__ == "__main__":
    plac.call(main)

