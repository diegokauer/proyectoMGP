import math
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import time


class SimpleNet(nn.Module):
    def __init__(self, in_channels, embedding_dim):
        super().__init__()

        def get_timestep_embedding(timesteps, embedding_dim):
            """
            Embeddings posicionales sinusoidales
            Fuente: https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
            """
            assert len(timesteps.size()) == 1

            half_dim = embedding_dim // 2
            emb = torch.arange(half_dim, dtype=torch.float32).cuda()
            emb = torch.exp(emb * (-math.log(10000) / (half_dim - 1))).cuda()
            emb = timesteps.view(-1, 1) * emb.view(1, -1).cuda()
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).cuda()

            if embedding_dim % 2 == 1:  # zero pad
                emb = F.pad(emb, (0, 1))

            assert emb.shape == (timesteps.size(0), embedding_dim)
            return emb

        self.embedding_dim = embedding_dim
        self. ts_embedding = get_timestep_embedding
        self.blocks = nn.Sequential(
            nn.Linear(in_channels + embedding_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, in_channels),
        )

    def forward(self, inputs, time_step):
        emb = self.ts_embedding(time_step, self.embedding_dim)
        emb = torch.cat([emb for _ in range(inputs.shape[0])])
        inputs_emb = torch.cat((inputs, emb), 1)
        outputs = self.blocks(inputs_emb)
        return outputs


class ScoreModel(object):
    def __init__(self, encoder, ndim, embedding_dim, nsigmas, device, lr):
        self.model = SimpleNet(ndim, embedding_dim).to(device)
        self.encoder = encoder.encode
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.sigmas = np.geomspace(5, 0.01, num=nsigmas)

    def noise_conditional_loss(self, input, sigmas):
        loss = 0
        net = self.model
        for i, sigma in enumerate(sigmas):
            samples = input.shape[0]
            x = input + torch.normal(0, sigma, size=input.shape).cuda()
            score = net(x.cuda(), torch.tensor([i]).cuda())
            nabla_log = (x - input) / (sigma ** 2)
            l = (score.view(samples, -1) + nabla_log.view(samples, -1)) ** 2
            l = l.sum(dim=-1).mean(dim=0)
            loss = l * 0.5 * sigma ** 2 + loss
        loss = loss / len(sigmas)
        return loss

    def train(self, epochs, data_loader):
        best_loss = 1e10
        loss_epoch = 0
        for epoch in range(epochs):
            start_time = time.time()
            for image, labels in data_loader:
                self.optimizer.zero_grad()
                x = self.encoder(image.to(self.device))
                # Calculamps la pérdida utilizando denoising_score_loss
                loss = self.noise_conditional_loss(torch.flatten(x, 1), self.sigmas)
                # Backpropagation y actualización de los pesos
                loss.backward()
                self.optimizer.step()

                loss_epoch += loss.item()

            avg_loss = loss_epoch / len(data_loader)
            time_epoch = time.time() - start_time
            print(f'Época [{epoch}/{epochs}], Pérdida: {avg_loss}')

            if best_loss >= avg_loss:
                best_loss = avg_loss
                self.save()



    def save(self):
        torch.save(self.model.state_dict(), '../checkpoint/score_network')

    def load(self):
        self.model.load_state_dict(
            torch.load('../checkpoint/score_network', map_location=torch.device(self.device))
        )
        print('Modelo Cargado')