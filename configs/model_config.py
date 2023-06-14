from models.pre_activation_block import PreActivationBlock
from models.pre_activation_bottleneck import PreActivationBottleneck

RESNET_CONFIGS = {
    '18': [[2, 2, 2, 2], PreActivationBlock],
    '28': [[3, 4, 6, 3], PreActivationBlock],
    '34': [[3, 4, 6, 3], PreActivationBlock],
    '50': [[3, 4, 6, 3], PreActivationBottleneck],
    '101': [[3, 4, 23, 3], PreActivationBottleneck]
}
