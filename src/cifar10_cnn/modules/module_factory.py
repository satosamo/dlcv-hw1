import torch
import torch.nn as nn


ACTIVATIONS = {
    "relu": lambda: nn.ReLU(),
    "leakyrelu03": lambda: nn.LeakyReLU(0.3),
    "leakyrelu01": lambda: nn.LeakyReLU(0.1),
    "leakyrelu003": lambda: nn.LeakyReLU(0.03),
    "gelu": lambda: nn.GELU(),
    "silu": lambda: nn.SiLU(),
    "tanh": lambda: nn.Tanh(),
    "prelu": lambda: nn.PReLU(),
    "sigmoid": lambda: nn.Sigmoid(),
    "elu": lambda: nn.ELU(),
}

NORMS1D = {
    "batchnorm1d": nn.BatchNorm1d,
    "layernorm": nn.LayerNorm,
    None: None
}

NORMS2D = {
    "batchnorm2d": nn.BatchNorm2d,
    "groupnorm": nn.GroupNorm,
    None: None
}


class FeaturesBlock(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size_conv, 
        padding, 
        activation, 
        kernel_size_pool, 
        norm, 
        dropout=0.0, 
        residual=False
    ):
        super().__init__()

        self.conv2 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size_conv, 
            padding=padding
        )
        
        self.act = activation()
        
        if norm is None:
            self.norm = nn.Identity()
        else:
            self.norm = norm(out_channels)
        
        if kernel_size_pool is None:
            self.pool = nn.Identity()
        else:
            self.pool = nn.MaxPool2d(kernel_size_pool)
        
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        
        self.residual = residual and (in_channels == out_channels) and (kernel_size_pool is None)


    def forward(self, x):
        out = self.conv2(x)
        out = self.act(out)
        out = self.norm(out)
        out = self.pool(out)
        out = self.dropout(out)

        if self.residual:
            out = out + x

        return out


class ClassifierBlock(nn.Module):
    def __init__(
        self, 
        in_dim, 
        out_dim, 
        activation, 
        norm, 
        dropout=0.0, 
        residual=False
    ):
        super().__init__()

        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = activation()
        self.residual = residual and (in_dim == out_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        if norm is None:
            self.norm = nn.Identity()
        else:
            self.norm = norm(out_dim)

    def forward(self, x):
        out = self.linear(x)
        out = self.activation(out)
        out = self.norm(out)
        out = self.dropout(out)

        if self.residual:
            out = out + x

        return out

class Cifar10_CNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        kernel_size_conv = cfg.get("kernel_size_conv")
        padding = cfg.get("padding")
        activation_feat = ACTIVATIONS[cfg["activation_feat"]]
        norm_feat = NORMS2D[cfg["norm_feat"]]
        pool_every = cfg.get("pool_every", 1)
        kernel_size_pool = cfg.get("kernel_size_pool")
        dropout_feat= cfg.get("dropout_feat", 0.0)
        residual_feat= cfg.get("residual_feat", False)

        activation_cls = ACTIVATIONS[cfg["activation_cls"]]
        norm_cls = NORMS1D[cfg["norm_cls"]]
        dropout_cls = cfg.get("dropout", 0.0)
        residual_cls = cfg.get("residual", False)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        layers_feat = []
        prev = cfg["in_channels"]

        for i, c in enumerate(cfg["hidden_feat_out_channels"]):
            layers_feat.append(
                FeaturesBlock(
                    prev, c,
                    kernel_size_conv=kernel_size_conv,
                    padding=padding,
                    activation=activation_feat,
                    kernel_size_pool=kernel_size_pool if ((i + 1) % pool_every == 0) else None,
                    norm=norm_feat,
                    dropout=dropout_feat,
                    residual=residual_feat
                )
            )
            prev = c

        self.features = nn.Sequential(*layers_feat)

        datadim = torch.zeros(1, cfg["in_channels"], 32, 32)
        with torch.no_grad():
            datadim = self.features(datadim)
            datadim = self.avgpool(datadim)
            flatten_dim = datadim.numel()

        layers_cls = []
        prev = flatten_dim

        for d in cfg["hidden_cls_out_dims"]:
            layers_cls.append(
                ClassifierBlock(
                    prev, d,
                    activation=activation_cls,
                    norm=norm_cls,
                    dropout=dropout_cls,
                    residual=residual_cls
                )
            )
            prev = d

        # final layer
        layers_cls.append(nn.Linear(prev, cfg["output_dim"]))

        self.classifier = nn.Sequential(*layers_cls)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
