import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator_(nn.Module):
    """
    Implementation of a simple GAN discriminator.

    Initialisation input:
        - latent_dim (int): Dimention of the latent space (Ex: 100)
        - out_shape (tuple): Shape of the output targeted data (Ex: (10, 10))
    """

    def __init__(self, latent_dim, out_shape, n_layers=4, n_units=512):
        super(Generator_, self).__init__()
        self.out_shape = out_shape
        if type(out_shape) is tuple and len(out_shape) > 1:
            out = int(np.prod(out_shape))
            self.tag = 1
        else:
            self.tag = 0
            out = int(out_shape)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU())
            return layers

        modules = nn.Sequential(*block(latent_dim, n_units, normalize=False))

        for i in range(n_layers):
            modules.add_module('linear_{}'.format(i), nn.Linear(n_units, n_units))
            modules.add_module('leakyRel_{}'.format(i), nn.LeakyReLU())

        modules.add_module('linear_{}'.format(n_layers + 1), nn.Linear(n_units, out))
        self.model = modules

    def forward(self, z):
        out = self.model(z)
        if self.tag:
            out = out.view(out.size(0), *self.out_shape)
        return out


class Discriminator_(nn.Module):
    """
    Implementation of a simple GAN sicriminator.

    Initialisation input:
        - inp_shape (tuple) : Tuple representing the shape of the inputs (Ex: (10, 10)
    """

    def __init__(self, inp_shape, n_layers=4, n_units=512):
        super(Discriminator_, self).__init__()

        if type(inp_shape) is tuple and len(inp_shape) > 1:
            inp = int(np.prod(inp_shape))
            self.tag = 1
        else:
            self.tag = 0
            inp = int(inp_shape)

        modules = nn.Sequential()
        modules.add_module('linear', nn.Linear(inp, n_units))
        modules.add_module('leakyRelu', nn.LeakyReLU())
        for i in range(n_layers):
            modules.add_module('linear_{}'.format(i), nn.Linear(n_units, n_units))
            modules.add_module('tanh_{}'.format(i), nn.Tanh())

        modules.add_module('linear_{}'.format(n_layers + 1), nn.Linear(n_units, 1))
        self.model = modules

    def forward(self, img):
        if self.tag:
            img = img.view(img.size(0), -1)
        validity = self.model(img)

        return validity


class ResNet(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs


class RoundtripModel(object):
    """
    Implementation of a Roundtrip model from the Paper : 'Density estimation using deep generative neural networks' by
    Qiao Liu, Jiaze Xu, Rui Jiang, Wing Hung Wong

    Initialisation inputs:
        - g_net (nn.Module): The first Generator model G
        - h_net (nn.Module): The second Generator model H
        - dx_net (nn.Module): The first Discriminator
        - dy_net (nn.Module): The second Discriminator
        - data (string): Name of the data set used (for name of the saved model)
        - x_sampler : A python method implementing a sampler from the complex distribution
        - y_sampler : A python method implementing a sampler from lattent distribution
        - batch_size (int) : Siez of the batches for training
    """

    def __init__(self, encoder_model, x_sampler, g_net, h_net, dx_net, dy_net, alpha, beta, batch_size, device='cpu'):
        self.encoder_model = encoder_model.to(device)
        self.x_sampler = x_sampler
        self.g_net = g_net.to(device)
        self.h_net = h_net.to(device)
        self.dx_net = dx_net.to(device)
        self.dy_net = dy_net.to(device)
        self.alpha = alpha
        self.beta = beta
        self.device = device
        self.batch_size = batch_size

        self.g_h_optim = torch.optim.Adam(list(self.g_net.parameters()) + list(self.h_net.parameters()), lr=2e-4,
                                          betas=(0.5, 0.9))
        self.d_optim = torch.optim.Adam(list(self.dx_net.parameters()) + list(self.dy_net.parameters()), lr=2e-4,
                                        betas=(0.5, 0.9))

    def discriminators_loss(self, x, y):
        fake_y = self.g_net(x)
        fake_x = self.h_net(y)

        dx = self.dx_net(x)
        dy = self.dy_net(y)

        d_fake_x = self.dx_net(fake_x)
        d_fake_y = self.dy_net(fake_y)
        # (1-d(x))^2
        dx_loss = (torch.mean((0.9 * torch.ones_like(dx) - dx) ** 2) + torch.mean(
            (0.1 * torch.ones_like(d_fake_x) - d_fake_x) ** 2)) / 2.0
        dy_loss = (torch.mean((0.9 * torch.ones_like(dy) - dy) ** 2) + torch.mean(
            (0.1 * torch.ones_like(d_fake_y) - d_fake_y) ** 2)) / 2.0
        d_loss = dx_loss + dy_loss
        return dx_loss, dy_loss, d_loss

    def generators_loss(self, x, y):
        y_ = self.g_net(x)
        x_ = self.h_net(y)

        x__ = self.h_net(y_)
        y__ = self.g_net(x_)

        dy_ = self.dy_net(y_)
        dx_ = self.dx_net(x_)

        l2_loss_x = torch.mean((x - x__) ** 2)
        l2_loss_y = torch.mean((y - y__) ** 2)

        # (1-d(x))^2
        g_loss_adv = torch.mean((0.9 * torch.ones_like(dy_) - dy_) ** 2)
        h_loss_adv = torch.mean((0.9 * torch.ones_like(dx_) - dx_) ** 2)

        g_loss = g_loss_adv + self.alpha * l2_loss_x + self.beta * l2_loss_y
        h_loss = h_loss_adv + self.alpha * l2_loss_x + self.beta * l2_loss_y
        g_h_loss = g_loss_adv + h_loss_adv + self.alpha * l2_loss_x + self.beta * l2_loss_y
        return g_loss, h_loss, g_h_loss

    def train(self, epochs, data_loader):
        best_loss = 1e10
        loss_epoch = 0
        for epoch in range(epochs):
            start_time = time.time()
            for image, labels in data_loader:
                bx = self.x_sampler.get_batch(image.shape[0])
                image = image.to(self.device)

                self.g_h_optim.zero_grad()
                self.d_optim.zero_grad()

                with torch.no_grad():
                    y = self.encoder_model.encoder(image)  # latent

                # y = torch.flatten(y, 1)#.to(torch.float32)
                labels = F.one_hot(labels).to(self.device).to(torch.float32)
                x = torch.Tensor(bx).to(self.device)
                x = torch.cat((x, labels), 1)

                dx_loss, dy_loss, d_loss = self.discriminators_loss(x, y)
                g_loss, h_loss, g_h_loss = self.generators_loss(x, y)

                d_loss.backward()
                g_h_loss.backward()
                self.g_h_optim.step()
                self.d_optim.step()

                loss_epoch += d_loss.item() + g_h_loss.item()

            avg_loss = loss_epoch / len(data_loader)
            time_epoch = time.time() - start_time
            print(
                f'Epoch: [{epoch}], Time: [{time_epoch:5.4f}], g_h_loss: [{g_h_loss:.4f}], d_loss: [{d_loss:.4f}], ydx_loss: [{dx_loss:.4f}], dy_loss: [{dy_loss:.4f}], g_loss: [{g_loss:.4f}], h_loss: [{h_loss:.4f}], avg_loss: [{avg_loss:.4f}]')

            if best_loss > avg_loss:
                best_loss = avg_loss
                self.save()

    def predict_y(self, x_point):
        bx = self.x_sampler.get_batch(x_point.shape[0])
        x = torch.Tensor(bx).to(self.device)
        x = torch.cat((x, x_point), 1)
        y = self.g_net(x).reshape(x.shape[0], 516, 8, 8)
        return self.encoder_model.codebook.straight_through(y)

    def predict_x(self, y_point):
        return self.h_net(y_point)

    def save(self, path=None):
        if path == None:
            torch.save(self.g_net.state_dict(), './checkpoint/g_net_CIFAR10')
            torch.save(self.h_net.state_dict(), './checkpoint/h_net_CIFAR10')
            torch.save(self.dx_net.state_dict(), './checkpoint/dx_net_CIFAR10')
            torch.save(self.dy_net.state_dict(), './checkpoint/dy_net_CIFAR10')
        else:
            torch.save(self.g_net.state_dict(), path)
            torch.save(self.h_net.state_dict(), path)
            torch.save(self.dx_net.state_dict(), path)
            torch.save(self.dy_net.state_dict(), path)

    def load(self, path=None):
        if path == None:
            self.g_net.load_state_dict(
                torch.load('./checkpoint/g_net_CIFAR10', map_location=torch.device(self.device)))
            self.h_net.load_state_dict(
                torch.load('./checkpoint/h_net_CIFAR10', map_location=torch.device(self.device)))
            self.dx_net.load_state_dict(
                torch.load('./checkpoint/dx_net_CIFAR10', map_location=torch.device(self.device)))
            self.dy_net.load_state_dict(
                torch.load('./checkpoint/dy_net_CIFAR10', map_location=torch.device(self.device)))
        else:
            self.g_net.load_state_dict(
                torch.load(path, map_location=torch.device(self.device)))
            self.h_net.load_state_dict(
                torch.load(path, map_location=torch.device(self.device)))
            self.dx_net.load_state_dict(
                torch.load(path, map_location=torch.device(self.device)))
            self.dy_net.load_state_dict(
                torch.load(path, map_location=torch.device(self.device)))
        print('Modelo Cargado')
