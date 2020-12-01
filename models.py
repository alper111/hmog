import os
import time

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

from blocks import HMOGBlock, ConvEncoder
from utils import gradient_penalty, nn_accuracy, FID_score


class HMOG:
    def __init__(self, opts):
        self.z_dim = opts["z_dim"]
        self.device = opts["device"]
        self.out_dir = opts["out_dir"]
        if os.path.exists(self.out_dir):
            for item in os.listdir(self.out_dir):
                os.remove(os.path.join(self.out_dir, item))
            os.rmdir(self.out_dir)
        os.makedirs(self.out_dir)

        self.G = HMOGBlock(opts["g_layers"], opts["g_input_shape"], opts["z_dim"], opts["depth"],
                           projection="linear", activation=torch.nn.ReLU(), std=0.02, normalization=None)
        self.D = ConvEncoder(opts["d_layers"], opts["d_input_shape"], z_dim=1,
                             activation=torch.nn.LeakyReLU(0.2), std=0.02,
                             normalization="layer_norm")

        self.G.to(self.device)
        self.D.to(self.device)

        self.G_optim = torch.optim.Adam(params=self.G.parameters(), lr=opts["lr"], betas=(0.5, 0.999), amsgrad=True)
        self.D_optim = torch.optim.Adam(params=self.D.parameters(), lr=opts["lr"], betas=(0.5, 0.999), amsgrad=True)

    def train(self, trainloader, testloader, inception, N, batch_size, epoch,
              c_iter=5, topk=5, acc=True, test_size=1000, test_step=5, img_step=5):

        fid_total = []
        real_acc_total = []
        fake_acc_total = []
        gen_loss_total = []
        disc_loss_total = []

        loop_per_epoch = N // (batch_size * c_iter)
        for e in range(epoch):
            gen_avg_loss = 0.0
            disc_avg_loss = 0.0
            start = time.time()
            iterator = iter(trainloader)

            for i in tqdm(range(loop_per_epoch)):
                for c in range(c_iter):
                    self.D_optim.zero_grad()
                    # train D with real data
                    x_real, _ = iterator.next()
                    x_real = x_real.to(self.device)
                    d_real_loss = -self.D(x_real).mean()

                    # train D with fake data
                    z = torch.randn(batch_size, self.z_dim, device=self.device)
                    x_fake = self.G(z)
                    d_fake_loss = self.D(x_fake).mean()

                    # gradient penalty loss to satisfy Lipschitz condition
                    d_grad_loss = gradient_penalty(self.D, x_real, x_fake, 1.0, self.device)

                    d_loss = d_real_loss + d_fake_loss + d_grad_loss
                    d_loss.backward()
                    self.D_optim.step()
                    disc_avg_loss += d_loss.item()

                # train G
                for p in self.D.parameters():
                    p.requires_grad = False
                self.G_optim.zero_grad()

                z = torch.randn(batch_size, self.z_dim, device=self.device)
                x_fake = self.G(z)
                g_loss = -self.D(x_fake).mean()

                g_loss.backward()
                self.G_optim.step()
                gen_avg_loss += g_loss.item()

                for p in self.D.parameters():
                    p.requires_grad = True

            finish = time.time()
            time_elapsed = finish - start

            gen_loss_total.append(gen_avg_loss/loop_per_epoch)
            disc_loss_total.append(disc_avg_loss/(loop_per_epoch*c_iter))
            print("Epoch: %d\tdisc loss: %.5f\tgen loss: %.5f\ttime elapsed: %.3f"
                  % (e+1, gen_loss_total[-1], disc_loss_total[-1], time_elapsed))

            if (e+1) == 1:
                eta = time_elapsed * epoch
                finish = time.asctime(time.localtime(time.time()+eta))
                print("### set your alarm at:", finish, "###")

            # save sample images
            if (e+1) % img_step == 0:
                self.save_images(n=100, filename=os.path.join(self.out_dir, "{0}.png".format(e+1)))

            # test
            if (e+1) % test_step == 0:
                fid, nn_real, nn_fake = self.test(testloader, test_size, inception)
                fid_total.append(fid)
                real_acc_total.append(nn_real)
                fake_acc_total.append(nn_fake)
                print("FID: %.5f\tReal acc: %.5f\tFake acc: %.5f" % (fid, nn_real, nn_fake))

            # save statistics
            if not os.path.exists(self.out_dir):
                os.makedirs(self.out_dir)

            self.G.eval()
            self.D.eval()
            np.save(os.path.join(self.out_dir, "fid.npy"), fid_total)
            np.save(os.path.join(self.out_dir, "ra.npy"), real_acc_total)
            np.save(os.path.join(self.out_dir, "fa.npy"), fake_acc_total)
            np.save(os.path.join(self.out_dir, "gloss.npy"), gen_loss_total)
            np.save(os.path.join(self.out_dir, "dloss.npy"), disc_loss_total)
            torch.save(self.G.cpu().state_dict(), os.path.join(self.out_dir, "g.pth"))
            torch.save(self.D.cpu().state_dict(), os.path.join(self.out_dir, "d.pth"))
            self.G.to(self.device)
            self.D.to(self.device)
            self.G.train()
            self.D.train()

        plt.plot(fake_acc_total)
        plt.plot(real_acc_total)
        plt.plot((np.array(fake_acc_total)+np.array(real_acc_total)) * 0.5, "--")
        plt.legend(["fake acc.", "real acc.", "total acc."])
        pp = PdfPages(os.path.join(self.out_dir, "accuracy.pdf"))
        pp.savefig()
        pp.close()
        plt.close()

        plt.plot(disc_loss_total)
        plt.plot(gen_loss_total)
        plt.legend(["disc. loss", "gen. loss"])
        pp = PdfPages(os.path.join(self.out_dir, "loss.pdf"))
        pp.savefig()
        pp.close()
        plt.close()

        plt.plot(fid_total)
        pp = PdfPages(os.path.join(self.out_dir, "fid.pdf"))
        pp.savefig()
        pp.close()
        plt.close()

    def test(self, testloader, test_size, inception=None):
        with torch.no_grad():
            self.G.eval()
            self.D.eval()
            x_real, _ = iter(testloader).next()
            x_real = x_real.to(self.device)
            x_fake = []
            if inception:
                f_real = []
                f_fake = []
            for i in range(test_size // 100):
                z = torch.randn(100, self.z_dim, device=self.device)
                x_i = self.G(z) * 0.5 + 0.5
                x_fake.append(x_i)
                if inception:
                    f_real.append(inception(x_real[i*100:(i+1)*100]))
                    f_fake.append(inception(x_i))
            x_fake = torch.cat(x_fake, dim=0)
            if inception:
                f_real = torch.cat(f_real, dim=0)
                f_fake = torch.cat(f_fake, dim=0)
                fid = FID_score(f_real.cpu(), f_fake.cpu())
                nn_real, nn_fake = nn_accuracy(f_real, f_fake, device=self.device)
            else:
                fid = -1
                nn_real, nn_fake = nn_accuracy(x_real, x_fake, device=self.device)
        return fid, nn_real, nn_fake

    def save_images(self, n, filename, nrow=10):
        self.G.eval()
        with torch.no_grad():
            z = torch.randn(n, self.z_dim, device=self.device)
            samples = self.G(z) * 0.5 + 0.5
            torchvision.utils.save_image(samples, filename, nrow=nrow)
        self.G.train()
