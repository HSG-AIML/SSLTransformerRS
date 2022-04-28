import torch
import torch.nn as nn
import torchvision.models as models

from exceptions import InvalidBackboneError


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}

        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        return self.backbone(x)
    
class NormalSimCLRDownstream(nn.Module):
    def __init__(self, base_model, out_dim, checkpoint, num_classes, load_late=False):
        super(NormalSimCLRDownstream, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}

        self.backbone = self._get_basemodel(base_model)
        self.dim_mlp = self.backbone.fc.in_features
        self.checkpoint = checkpoint
        self.num_classes = num_classes
        self.load_late = load_late
        
        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(self.dim_mlp, self.dim_mlp), nn.ReLU(), self.backbone.fc)
        
        self.init_and_load_trained_state_dict()

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        return self.backbone(x)
    
    def init_and_load_trained_state_dict(self):
        """load the pre-trained backbone weights"""
        
        if not self.load_late:
            log_bb = self.load_state_dict(self.checkpoint["state_dict"], strict=True)
        
        # replace the projection MLP with a classification layer
        self.backbone.fc = torch.nn.Linear(self.dim_mlp, self.num_classes)
        
        if self.load_late:
            log_bb = self.load_state_dict(self.checkpoint["model_weights"], strict=True)

        # freeze all layers but the last fc
        for name, param in self.named_parameters():
            if name not in ['backbone.fc.weight', 'backbone.fc.bias']:
                param.requires_grad = False

class DoubleResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(DoubleResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18,
                            "resnet50": models.resnet50,}
        
        self.backbone1 = self.resnet_dict.get(base_model)(pretrained=False, num_classes=out_dim)
        self.backbone2 = self.resnet_dict.get(base_model)(pretrained=False, num_classes=out_dim)

        dim_mlp1 = self.backbone1.fc.in_features
        dim_mlp2 = self.backbone2.fc.in_features
        
        # add mlp projection head
        self.backbone1.fc = nn.Sequential(nn.Linear(dim_mlp1, dim_mlp1), nn.ReLU(), self.backbone1.fc)
        self.backbone2.fc = nn.Sequential(nn.Linear(dim_mlp2, dim_mlp2), nn.ReLU(), self.backbone2.fc)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        return {"s1" : self.backbone1(x["s1"]), "s2" : self.backbone2(x["s2"])}
    
    
class DoubleResNetSimCLRDownstream(nn.Module):
    """concatenate outputs from two backbones and add one linear layer"""

    def __init__(self, base_model, out_dim):
        super(DoubleResNetSimCLRDownstream, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18,
                            "resnet50": models.resnet50,}
        
        self.backbone1 = self.resnet_dict.get(base_model)(pretrained=False, num_classes=out_dim)
        self.backbone2 = self.resnet_dict.get(base_model)(pretrained=False, num_classes=out_dim)

        dim_mlp1 = self.backbone1.fc.in_features
        dim_mlp2 = self.backbone2.fc.in_features
        
        # add final linear layer
        self.fc = nn.Linear(dim_mlp1 + dim_mlp2, out_dim, bias=True)
        self.backbone1.fc = nn.Identity()
        self.backbone2.fc = nn.Identity()

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        x1 = self.backbone1(x["s1"])
        x2 = self.backbone2(x["s2"])
        
        z = torch.cat([x1, x2], dim=1)
        z = self.fc(z)
        
        return z
    
    def load_trained_state_dict(self, weights):
        """load the pre-trained backbone weights"""
        
        # remove the MLP projection heads
        for k in list(weights.keys()):
            if k.startswith(('backbone1.fc', 'backbone2.fc')):
                del weights[k]
        
        log = self.load_state_dict(weights, strict=False)
        assert log.missing_keys == ['fc.weight', 'fc.bias']
        
        # freeze all layers but the last fc
        for name, param in self.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False
