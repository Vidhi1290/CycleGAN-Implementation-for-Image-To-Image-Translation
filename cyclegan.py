# Code credit : PyTorch-GAN Git repo

if __name__ == '__main__':
    import argparse
    import os
    import numpy as np
    import math
    import itertools
    import datetime
    import time

    import torchvision.transforms as transforms
    from torchvision.utils import save_image, make_grid

    from torch.utils.data import DataLoader
    from torchvision import datasets
    from torch.autograd import Variable

    from models import *
    from datasets import *
    from utils import *

    import torch.nn as nn
    import torch.nn.functional as F
    import torch
    torch.cuda.empty_cache()
    epoch = 0
    n_epochs = 20
    dataset_name = "horse2zebra"
    batch_size = 2
    lr = 0.0002
    b1 = 0.5
    b2 = 0.0002
    decay_epoch = 10
    n_cpu = 4
    img_height,img_width,channels=256,256,3
    sample_interval = 100
    n_residual_blocks = 9
    lambda_cyc = 10
    lambda_id = 5.0
    checkpoint_interval = 1
    
    # Create sample and checkpoint directories
    os.makedirs("images/%s" % dataset_name, exist_ok=True)
    os.makedirs("saved_models/%s" % dataset_name, exist_ok=True)

    # Losses
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    cuda = torch.cuda.is_available()

    input_shape = (channels, img_height, img_width)

    # Initialize generator and discriminator
    G_AB = GeneratorResNet(input_shape, n_residual_blocks)
    G_BA = GeneratorResNet(input_shape, n_residual_blocks)
    D_A = Discriminator(input_shape)
    D_B = Discriminator(input_shape)

    if cuda:
        G_AB = G_AB.cuda()
        G_BA = G_BA.cuda()
        D_A = D_A.cuda()
        D_B = D_B.cuda()
        criterion_GAN.cuda()
        criterion_cycle.cuda()
        criterion_identity.cuda()

    if epoch != 0:
        # Load pretrained models
        G_AB.load_state_dict(torch.load("saved_models/%s/G_AB_%d.pth" % (dataset_name, epoch)))
        G_BA.load_state_dict(torch.load("saved_models/%s/G_BA_%d.pth" % (dataset_name, epoch)))
        D_A.load_state_dict(torch.load("saved_models/%s/D_A_%d.pth" % (dataset_name, epoch)))
        D_B.load_state_dict(torch.load("saved_models/%s/D_B_%d.pth" % (dataset_name, epoch)))
    else:
        # Initialize weights
        G_AB.apply(weights_init_normal)
        G_BA.apply(weights_init_normal)
        D_A.apply(weights_init_normal)
        D_B.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=lr, betas=(b1, b2)
    )
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=lr, betas=(b1, b2))

    # Learning rate update schedulers
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step
    )
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_A, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step
    )
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_B, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step
    )

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    # Buffers of previously generated samples
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # Image transformations
    transforms_ = [
        transforms.Resize(int(img_height * 1.12), Image.BICUBIC),
        transforms.RandomCrop((img_height, img_width)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    # Training data loader
    dataloader = DataLoader(
        ImageDataset("data/%s" % dataset_name, transforms_=transforms_, unaligned=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu,
    )
    # Test data loader
    val_dataloader = DataLoader(
        ImageDataset("data/%s" % dataset_name, transforms_=transforms_, unaligned=True, mode="test"),
        batch_size=5,
        shuffle=True,
        num_workers=1,
    )


    def sample_images(batches_done):
        """Saves a generated sample from the test set"""
        imgs = next(iter(val_dataloader))
        G_AB.eval()
        G_BA.eval()
        real_A = Variable(imgs["A"].type(Tensor))
        fake_B = G_AB(real_A)
        real_B = Variable(imgs["B"].type(Tensor))
        fake_A = G_BA(real_B)
        # Arange images along x-axis
        real_A = make_grid(real_A, nrow=5, normalize=True)
        real_B = make_grid(real_B, nrow=5, normalize=True)
        fake_A = make_grid(fake_A, nrow=5, normalize=True)
        fake_B = make_grid(fake_B, nrow=5, normalize=True)
        # Arange images along y-axis
        image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
        save_image(image_grid, "images/%s/%s.png" % (dataset_name, batches_done), normalize=False)


    # ----------
    #  Training
    # ----------

    prev_time = time.time()
    for epoch in range(epoch, n_epochs):
        for i, batch in enumerate(dataloader):

            # Set model input
            real_A = Variable(batch["A"].type(Tensor))
            real_B = Variable(batch["B"].type(Tensor))

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((real_A.size(0), *D_A.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A.output_shape))), requires_grad=False)

            # ------------------
            #  Train Generators
            # ------------------

            G_AB.train()
            G_BA.train()

            optimizer_G.zero_grad()

            # Identity loss
            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)

            loss_identity = (loss_id_A + loss_id_B) / 2

            # GAN loss
            fake_B = G_AB(real_A)
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
            fake_A = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)

            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # Cycle loss
            recov_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(recov_A, real_A)
            recov_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)

            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # Total loss
            loss_G = loss_GAN + lambda_cyc * loss_cycle + lambda_id * loss_identity

            loss_G.backward()
            optimizer_G.step()

            # -----------------------
            #  Train Discriminator A
            # -----------------------

            optimizer_D_A.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_A(real_A), valid)
            # Fake loss (on batch of previously generated samples)
            fake_A_ = fake_A_buffer.push_and_pop(fake_A)
            loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
            # Total loss
            loss_D_A = (loss_real + loss_fake) / 2

            loss_D_A.backward()
            optimizer_D_A.step()

            # -----------------------
            #  Train Discriminator B
            # -----------------------

            optimizer_D_B.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_B(real_B), valid)
            # Fake loss (on batch of previously generated samples)
            fake_B_ = fake_B_buffer.push_and_pop(fake_B)
            loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
            # Total loss
            loss_D_B = (loss_real + loss_fake) / 2

            loss_D_B.backward()
            optimizer_D_B.step()

            loss_D = (loss_D_A + loss_D_B) / 2

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s"
                % (
                    epoch,
                    n_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_GAN.item(),
                    loss_cycle.item(),
                    loss_identity.item(),
                    time_left,
                )
            )

            # If at sample interval save image
            if batches_done % sample_interval == 0:
                sample_images(batches_done)
                #sys.stdout.write(torch.cuda.memory_summary(device=None, abbreviated=False))

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(G_AB.state_dict(), "saved_models/%s/G_AB_%d.pth" % (opt.dataset_name, epoch))
            torch.save(G_BA.state_dict(), "saved_models/%s/G_BA_%d.pth" % (opt.dataset_name, epoch))
            torch.save(D_A.state_dict(), "saved_models/%s/D_A_%d.pth" % (opt.dataset_name, epoch))
            torch.save(D_B.state_dict(), "saved_models/%s/D_B_%d.pth" % (opt.dataset_name, epoch))
