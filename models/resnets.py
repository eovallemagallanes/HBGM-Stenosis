from timm.models.layers.create_attn import *
from functools import partial
import timm

size_output_layer3 = {'resnet18': 256,
                      'resnet34': 256,
                      'resnet50': 1024}

size_output_layer4 = {'resnet18': 512,
                      'resnet34': 512,
                      'resnet50': 2048}


def create_model(model_name, att_name, num_classes=1, rd_ration=0.25, spatial_kernel_size=3):
    if model_name in size_output_layer3:
        print('--> Creating ', model_name)
    else:
        assert False, "Invalid model name (%s)" % model_name

    if att_name=='None':
        print('--> ORIGINAL RESNET')
        net = timm.create_model(model_name, num_classes=num_classes)
        #net.fc = torch.nn.Linear(size_output_layer4[model_name], num_classes)
    elif att_name=='cbam':
        print('--> CBAM RESNET')
        attn_layer = partial(get_attn('cbam'), rd_ratio=rd_ration, spatial_kernel_size=spatial_kernel_size)
        block_args=dict(attn_layer=attn_layer)
        if model_name=='resnet18':
            net = timm.models.resnet.resnet18(block_args=block_args, num_classes=num_classes)
        elif model_name=='resnet34':
            net = timm.models.resnet.resnet34(block_args=block_args, num_classes=num_classes)
        elif model_name=='resnet50':
            net = timm.models.resnet.resnet50(block_args=block_args, num_classes=num_classes)
        else:
            assert False, "Invalid model name (%s)" % model_name

        # net.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        net.layer4 = torch.nn.Identity()
        net.fc = torch.nn.Linear(size_output_layer3[model_name], num_classes)

    return net