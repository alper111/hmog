import math

import numpy
import torch
import torchvision


class Linear(torch.nn.Module):
    '''linear layer with optional batch normalization or layer normalization'''
    def __init__(self, in_features, out_features, std=None, normalization=None):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        self.normalization = normalization
        if normalization == 'batch_norm':
            self.normalization_func = torch.nn.BatchNorm1d(num_features=self.out_features)
        elif normalization == 'layer_norm':
            self.normalization_func = torch.nn.LayerNorm(normalized_shape=self.out_features)

        if std is not None:
            self.weight.data.normal_(0., std)
            self.bias.data.normal_(0., std)
        else:
            # he initialization for ReLU activaiton
            stdv = math.sqrt(2 / self.weight.size(1))
            self.weight.data.normal_(0., stdv)
            self.bias.data.zero_()

    def forward(self, x):
        x = torch.nn.functional.linear(x, self.weight, self.bias)
        if self.normalization:
            x = self.normalization_func(x)
        return x

    def extra_repr(self):
        return 'in_features={}, out_features={}, normalization={}'.format(
            self.in_features, self.out_features, self.normalization
        )


class Conv2d(torch.nn.Module):
    '''convolutional layer with std option'''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, std=None, bias=True):
        super(Conv2d, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        if std is not None:
            self.conv.weight.data.normal_(0., std)
            if self.conv.bias is not None:
                self.conv.bias.data.normal_(0., std)

    def forward(self, x):
        x = self.conv(x)
        return x


class ConvTranspose2d(torch.nn.Module):
    '''convolution transpose layer with std option'''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, std=None, bias=True):
        super(ConvTranspose2d, self).__init__()
        self.convt = torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        if std is not None:
            self.convt.weight.data.normal_(0., std)
            if self.convt.bias is not None:
                self.convt.bias.data.normal_(0., std)

    def forward(self, x):
        x = self.convt(x)
        return x


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, std=None, bias=True, normalization=None, transposed=False):
        super(ConvBlock, self).__init__()
        if transposed:
            self.block = [torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
        else:
            self.block = [torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
        self.block.append(torch.nn.ReLU())
        if normalization == "batch_norm":
            self.block.append(torch.nn.BatchNorm2d(out_channels))

        if std is not None:
            self.block[0].weight.data.normal_(0., std)
            self.block[0].bias.data.normal_(0., std)
        self.block = torch.nn.Sequential(*self.block)

    def forward(self, x):
        return self.block(x)


class ConvEncoder(torch.nn.Module):
    '''DCGAN-like convolutional encoder'''
    def __init__(self, channels, input_shape, z_dim, activation=torch.nn.ReLU(), std=None, normalization=None):
        super(ConvEncoder, self).__init__()
        convolutions = []
        current_shape = input_shape
        for ch in channels:
            convolutions.append(Conv2d(in_channels=current_shape[0], out_channels=ch, kernel_size=4, stride=2, padding=1, std=std))
            current_shape = [ch, current_shape[1] // 2, current_shape[2] // 2]
            if normalization == 'batch_norm':
                convolutions.append(torch.nn.BatchNorm2d(ch))
            elif normalization == 'layer_norm':
                convolutions.append(torch.nn.LayerNorm(current_shape))
            convolutions.append(activation)
        self.convolutions = torch.nn.Sequential(*convolutions)
        self.dense = Linear(in_features=current_shape[0] * current_shape[1] * current_shape[2], out_features=z_dim, std=std, normalization=None)

    def forward(self, x):
        out = self.convolutions(x)
        out = out.reshape(out.shape[0], -1)
        out = self.dense(out)
        return out


class ConvDecoder(torch.nn.Module):
    '''DCGAN-like convolutional decoder'''
    def __init__(self, channels, input_shape, z_dim, activation=torch.nn.ReLU(), std=None, normalization=None):
        super(ConvDecoder, self).__init__()
        self.input_shape = input_shape
        self.dense = torch.nn.Sequential(
            Linear(in_features=z_dim, out_features=input_shape[0] * input_shape[1] * input_shape[2], std=std, normalization=normalization),
            activation)

        convolutions = []
        current_shape = input_shape
        for ch in channels[:-1]:
            convolutions.append(ConvTranspose2d(in_channels=current_shape[0], out_channels=ch, kernel_size=4, stride=2, padding=1, std=std))
            current_shape = [ch, current_shape[1] * 2, current_shape[2] * 2]
            if normalization == 'batch_norm':
                convolutions.append(torch.nn.BatchNorm2d(ch))
            elif normalization == 'layer_norm':
                convolutions.append(torch.nn.LayerNorm(current_shape))
            convolutions.append(activation)
        convolutions.append(ConvTranspose2d(in_channels=current_shape[0], out_channels=channels[-1], kernel_size=4, stride=2, padding=1, std=std))
        self.convolutions = torch.nn.Sequential(*convolutions)

    def forward(self, x):
        out = self.dense(x)
        out = out.reshape(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        out = self.convolutions(out)
        return out


class MLP(torch.nn.Module):
    def __init__(self, layer_info, activation, std=None, normalization=None):
        super(MLP, self).__init__()
        layers = []
        in_dim = layer_info[0]
        for j in layer_info[1:-1]:
            layers.append(Linear(in_features=in_dim, out_features=j, std=std, normalization=normalization))
            layers.append(activation)
            in_dim = j
        layers.append(Linear(in_features=in_dim, out_features=layer_info[-1], std=std, normalization=None))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class HME(torch.nn.Module):

    def __init__(self, in_features, out_features, depth, projection='linear'):
        super(HME, self).__init__()
        self.proj = projection
        self.depth = depth
        self.in_features = in_features
        self.out_features = out_features
        self.n_leaf = int(2**depth)
        self.gate_count = int(self.n_leaf - 1)
        self.gw = torch.nn.Parameter(
            torch.nn.init.kaiming_normal_(
                torch.empty(self.gate_count, in_features), nonlinearity='sigmoid').t())
        self.gb = torch.nn.Parameter(torch.zeros(self.gate_count))
        if self.proj == 'linear':
            self.pw = torch.nn.init.kaiming_normal_(torch.empty(out_features*self.n_leaf, in_features), nonlinearity='linear')
            self.pw = torch.nn.Parameter(self.pw.reshape(out_features, self.n_leaf, in_features).permute(0, 2, 1))
            self.pb = torch.nn.Parameter(torch.zeros(out_features, self.n_leaf))
        elif self.proj == 'constant':
            self.z = torch.nn.Parameter(torch.randn(out_features, self.n_leaf))

    def forward(self, x):
        node_densities = self.node_densities(x)
        leaf_probs = node_densities[:, -self.n_leaf:].t()

        if self.proj == 'linear':
            gated_projection = torch.matmul(self.pw, leaf_probs).permute(2, 0, 1)
            gated_bias = torch.matmul(self.pb, leaf_probs).permute(1, 0)
            result = torch.matmul(gated_projection, x.reshape(-1, self.in_features, 1))[:, :, 0] + gated_bias
        elif self.proj == 'constant':
            result = torch.matmul(self.z, leaf_probs).permute(1, 0)

        return result

    def node_densities(self, x):
        gatings = self.gatings(x)
        node_densities = torch.ones(x.shape[0], 2**(self.depth+1)-1, device=x.device)
        it = 1
        for d in range(1, self.depth+1):
            for i in range(2**d):
                parent_index = (it+1) // 2 - 1
                child_way = (it+1) % 2
                if child_way == 0:
                    parent_gating = gatings[:, parent_index]
                else:
                    parent_gating = 1 - gatings[:, parent_index]
                parent_density = node_densities[:, parent_index].clone()
                node_densities[:, it] = (parent_density * parent_gating)
                it += 1
        return node_densities

    def gatings(self, x):
        return torch.sigmoid(torch.add(torch.matmul(x, self.gw), self.gb))

    def total_path_value(self, z, index, level=None):
        gatings = self.gatings(z)
        gateways = numpy.binary_repr(index, width=self.depth)
        L = 0.
        current = 0
        if level is None:
            level = self.depth

        for i in range(level):
            if int(gateways[i]) == 0:
                L += gatings[:, current].mean()
                current = 2 * current + 1
            else:
                L += (1 - gatings[:, current]).mean()
                current = 2 * current + 2
        return L

    def extra_repr(self):
        return "in_features=%d, out_features=%d, depth=%d, projection=%s" % (
            self.in_features,
            self.out_features,
            self.depth,
            self.proj)


class ME(torch.nn.Module):

    def __init__(self, in_features, out_features, n_leaf, projection='linear', dropout=0.0):
        super(ME, self).__init__()
        self.proj = projection
        self.n_leaf = n_leaf
        self.in_features = in_features
        self.out_features = out_features
        self.gw = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(
                torch.empty(in_features, n_leaf)))
        self.gb = torch.nn.Parameter(torch.zeros(n_leaf))
        if self.proj == 'linear':
            self.pw = torch.nn.init.kaiming_normal_(torch.empty(out_features*n_leaf, in_features), nonlinearity='linear')
            self.pw = torch.nn.Parameter(self.pw.reshape(out_features, n_leaf, in_features).permute(0, 2, 1))
            self.pb = torch.nn.Parameter(torch.zeros(out_features, n_leaf))
        elif self.proj == 'constant':
            self.z = torch.nn.Parameter(torch.randn(out_features, n_leaf))

    def forward(self, x):
        gatings = torch.softmax(torch.add(torch.matmul(x, self.gw), self.gb), dim=1).t()
        if self.proj == 'linear':
            gated_projection = torch.matmul(self.pw, gatings).permute(2, 0, 1)
            gated_bias = torch.matmul(self.pb, gatings).permute(1, 0)
            result = torch.matmul(gated_projection, x.reshape(-1, self.in_features, 1))[:, :, 0] + gated_bias
        elif self.proj == 'constant':
            result = torch.matmul(self.z, gatings).permute(1, 0)
        return result

    def extra_repr(self):
        return "in_features=%d, out_features=%d, n_leaf=%d, projection=%s" % (
            self.in_features,
            self.out_features,
            self.n_leaf,
            self.proj)


class HMOGBlock(torch.nn.Module):
    def __init__(self, channels, input_shape, z_dim, depth, projection='linear', activation=torch.nn.ReLU(), std=None, normalization=None):
        super(HMOGBlock, self).__init__()
        self.input_shape = input_shape
        self.tree = HME(in_features=z_dim, out_features=input_shape[0] * input_shape[1] * input_shape[2], depth=depth, projection=projection)

        convolutions = []
        current_shape = input_shape
        for ch in channels[:-1]:
            convolutions.append(ConvTranspose2d(in_channels=current_shape[0], out_channels=ch, kernel_size=4, stride=2, padding=1, std=std))
            current_shape = [ch, current_shape[1] * 2, current_shape[2] * 2]
            if normalization == 'batch_norm':
                convolutions.append(torch.nn.BatchNorm2d(ch))
            elif normalization == 'layer_norm':
                convolutions.append(torch.nn.LayerNorm(current_shape))
            convolutions.append(activation)
        convolutions.append(ConvTranspose2d(in_channels=current_shape[0], out_channels=channels[-1], kernel_size=4, stride=2, padding=1, std=std))
        self.convolutions = torch.nn.Sequential(*convolutions)

    def forward(self, x):
        out = self.tree(x)
        out = out.reshape(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        out = self.convolutions(out)
        return out


class MOGBlock(torch.nn.Module):
    def __init__(self, channels, input_shape, z_dim, n_leaf, projection='linear', activation=torch.nn.ReLU(), std=None, normalization=None):
        super(MOGBlock, self).__init__()
        self.input_shape = input_shape
        self.mixture = ME(in_features=z_dim, out_features=input_shape[0] * input_shape[1] * input_shape[2], n_leaf=n_leaf, projection=projection)

        convolutions = []
        current_shape = input_shape
        for ch in channels[:-1]:
            convolutions.append(ConvTranspose2d(in_channels=current_shape[0], out_channels=ch, kernel_size=4, stride=2, padding=1, std=std))
            current_shape = [ch, current_shape[1] * 2, current_shape[2] * 2]
            if normalization == 'batch_norm':
                convolutions.append(torch.nn.BatchNorm2d(ch))
            elif normalization == 'layer_norm':
                convolutions.append(torch.nn.LayerNorm(current_shape))
            convolutions.append(activation)
        convolutions.append(ConvTranspose2d(in_channels=current_shape[0], out_channels=channels[-1], kernel_size=4, stride=2, padding=1, std=std))
        self.convolutions = torch.nn.Sequential(*convolutions)

    def forward(self, x):
        out = self.mixture(x)
        out = out.reshape(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        out = self.convolutions(out)
        return out


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

    def extra_repr(self):
        return 'identity function'


class InceptionV3(torch.nn.Module):
    def __init__(self):
        super(InceptionV3, self).__init__()
        self.inception = torchvision.models.inception_v3(pretrained=True)
        self.inception.eval()
        self.inception.fc = Identity()
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        if x.shape[1] != 3:
            x = x.repeat(1, 3, 1, 1)
        x = (x-self.mean)/self.std
        x = self.transform(x, mode='bilinear', size=(299, 299), align_corners=False)
        return self.inception(x)
