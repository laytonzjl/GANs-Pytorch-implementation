import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch import optim
from torch.utils.tensorboard import SummaryWriter


class Discriminator(nn.Module):
    def __init__(self, img_dim):  # 输入in_dim
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):  # 输入z_dim潜向量，输出img_dim图片
        super(Generator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, img_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.network(x)


device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
z_dim = 64
img_dim = 28 * 28 * 1
batch_size = 32
epochs = 20

disc = Discriminator(img_dim).to(device)
gen = Generator(z_dim, img_dim).to(device)

transforms = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))]
)
dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()
fake_img = SummaryWriter(f"log/fake/")
real_img = SummaryWriter(f"log/real/")
loss_img = SummaryWriter(f"log/loss/")
step = 0

for epoch in range(epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 28 * 28).to(device)
        batch_size = real.shape[0]

        noise = torch.randn((batch_size, z_dim)).to(device)
        fake = gen(noise)
        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()

        gen_fake = disc(fake).view(-1)
        lossG = criterion(gen_fake, torch.ones_like(gen_fake))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}] Batch {batch_idx}/{len(loader)} \
                              Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen(noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                fake_img.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                real_img.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                loss_img.add_scalar("Discriminator Loss", lossD.item(), global_step=step)
                loss_img.add_scalar("Generator Loss", lossG.item(), global_step=step)
                step += 1
