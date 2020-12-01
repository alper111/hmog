import sys

import yaml
import torch
import torchvision
import torchvision.transforms as transforms

from models import HMOG
from blocks import InceptionV3


opts = yaml.safe_load(open(sys.argv[1], "r"))
hmog = HMOG(opts)

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=opts["batch_size"], shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=opts["test_size"], num_workers=4)
# inception = InceptionV3().to(opts["device"])

hmog.train(trainloader, testloader, inception=None, len(trainset.targets),
           opts["batch_size"], opts["epoch"], test_size=opts["test_size"],
           test_step=opts["test_step"], img_step=opts["img_step"])
