import pdb
import copy
import utils
import torch
import types
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from modules.criterions import SeqKD
from modules import BiLSTMLayer, TemporalConv
from torchvision.models import get_model_weights



class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs


class SLRModel(nn.Module):
    def __init__(
            self, num_classes, c2d_type, conv_type, use_bn=False,
            hidden_size=1024, gloss_dict=None, loss_weights=None,
            weight_norm=True, share_classifier=True
    ):
        super(SLRModel, self).__init__()
        self.decoder = None
        self.loss = dict()
        self.criterion_init()
        self.num_classes = num_classes
        self.loss_weights = loss_weights

        weights_enum = get_model_weights(c2d_type)
        weights = weights_enum.DEFAULT if weights_enum else None
        # self.conv2d = getattr(models, c2d_type)(pretrained=True)
        self.conv2d = getattr(models, c2d_type)(weights=weights)

        # Handle MobileNetV2 specific output channels
        if c2d_type in ["mobilenet_v2", "mobilenet_v3_small"]:
            self.conv2d = self._modify_mobilenet(self.conv2d, c2d_type)
        else:  # Default case for ResNet
            self.conv2d.fc = Identity()

        if c2d_type == "squeezenet1_1":
            self.conv2d = self._modify_squeezenet(self.conv2d)
        if c2d_type == "efficientnet_b1":
            self.conv2d = self._modify_efficientnet(self.conv2d)
        

        self.conv1d = TemporalConv(input_size=512,
                                    hidden_size=hidden_size,
                                    conv_type=conv_type,
                                    use_bn=use_bn,
                                    num_classes=num_classes)
        self.decoder = utils.Decode(gloss_dict, num_classes, 'beam')
        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                            num_layers=2, bidirectional=True)
        if weight_norm:
            self.classifier = NormLinear(hidden_size, self.num_classes)
            self.conv1d.fc = NormLinear(hidden_size, self.num_classes)
        else:
            self.classifier = nn.Linear(hidden_size, self.num_classes)
            self.conv1d.fc = nn.Linear(hidden_size, self.num_classes)
        if share_classifier:
            self.conv1d.fc = self.classifier
        self.register_backward_hook(self.backward_hook)

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0

    def masked_bn(self, inputs, len_x):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        x = self.conv2d(x)
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                        for idx, lgt in enumerate(len_x)])
        return x

    def forward(self, x, len_x, label=None, label_lgt=None):
        if len(x.shape) == 5:
            # videos
            batch, temp, channel, height, width = x.shape
            inputs = x.reshape(batch * temp, channel, height, width)
            framewise = self.masked_bn(inputs, len_x)
            framewise = framewise.reshape(batch, temp, -1).transpose(1, 2)
        else:
            # frame-wise features
            framewise = x

        conv1d_outputs = self.conv1d(framewise, len_x)
        # x: T, B, C
        x = conv1d_outputs['visual_feat']
        lgt = conv1d_outputs['feat_len']
        tm_outputs = self.temporal_model(x, lgt)
        outputs = self.classifier(tm_outputs['predictions'])
        pred = None if self.training \
            else self.decoder.decode(outputs, lgt, batch_first=False, probs=False)
        conv_pred = None if self.training \
            else self.decoder.decode(conv1d_outputs['conv_logits'], lgt, batch_first=False, probs=False)

        return {
            "framewise_features": framewise,
            "visual_features": x,
            "feat_len": lgt,
            "conv_logits": conv1d_outputs['conv_logits'],
            "sequence_logits": outputs,
            "conv_sents": conv_pred,
            "recognized_sents": pred,
        }

    def criterion_calculation(self, ret_dict, label, label_lgt):
        loss = 0
        for k, weight in self.loss_weights.items():
            if k == 'ConvCTC':
                loss += weight * self.loss['CTCLoss'](ret_dict["conv_logits"].log_softmax(-1),
                                                        label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                        label_lgt.cpu().int()).mean()
            elif k == 'SeqCTC':
                loss += weight * self.loss['CTCLoss'](ret_dict["sequence_logits"].log_softmax(-1),
                                                        label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                        label_lgt.cpu().int()).mean()
            elif k == 'Dist':
                loss += weight * self.loss['distillation'](ret_dict["conv_logits"],
                                                            ret_dict["sequence_logits"].detach(),
                                                            use_blank=False)
        return loss

    def criterion_init(self):
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=False)
        self.loss['distillation'] = SeqKD(T=8)
        return self.loss

    def _modify_mobilenet(self, mobilenet, c2d_type):
        mobilenet.features = nn.Sequential(*mobilenet.features, nn.AdaptiveAvgPool2d((1, 1)))
        if c2d_type == "mobilenet_v3_small":
            in_features = 576
        elif c2d_type == "mobilenet_v3_large":
            in_features = 960
        else:
            in_features = 1280

        # Replace mobilenet's classifier to output 512 channels
        mobilenet.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512),  # Reduce output to 512 channels
            nn.ReLU(inplace=True)
        )
        return mobilenet

    def _modify_squeezenet(self, squeezenet):
        # Pooling untuk memastikan dimensi spasial tetap
        squeezenet.features = nn.Sequential(
            *squeezenet.features,
            nn.AdaptiveAvgPool2d((1, 1))  # Output spasial menjadi (1, 1)
        )
        # Linear layer untuk menyesuaikan jumlah channel output
        squeezenet.classifier = nn.Sequential(
            nn.Flatten(),  # Mengubah [Batch, C, 1, 1] menjadi [Batch, C]
            nn.Linear(512, 512),  # Atur output ke 512 channels
            nn.ReLU(inplace=True)
        )
        return squeezenet
    
    def _modify_efficientnet(self, efficientnet):
        # Pooling untuk mengurangi dimensi spasial ke (1, 1)
        efficientnet.features = nn.Sequential(
            *efficientnet.features,
            nn.AdaptiveAvgPool2d((1, 1))  # Mengubah dimensi spasial menjadi (1, 1)
        )
        # Linear layer untuk menyesuaikan jumlah channel
        efficientnet.classifier = nn.Sequential(
            nn.Flatten(),  # Mengubah [Batch, C, 1, 1] menjadi [Batch, C]
            nn.Linear(1280, 512),  # Sesuaikan output dari 1280 menjadi 512 channels
            nn.ReLU(inplace=True)
        )
        return efficientnet


