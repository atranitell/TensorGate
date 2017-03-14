
from tensorflow.contrib.framework import arg_scope
from nets import net_cifarnet
from nets import net_lenet
from nets import net_alexnet
from nets import net_resnet
from nets import net_vgg

networks_map = {
    'cifarnet': net_cifarnet.cifarnet(),
    'lenet': net_lenet.lenet(),
    'alexnet': net_alexnet.alexnet(),
    'resnet50': net_resnet.resnet50(),
    'resnet101': net_resnet.resnet101(),
    'resnet152': net_resnet.resnet152(),
    'resnet200': net_resnet.resnet200(),
    'vgg': net_vgg.vgg(),
    'vgg16': net_vgg.vgg16(),
    'vgg19': net_vgg.vgg19()
}


def check_network(name, data_type):
    if name not in networks_map:
        raise ValueError('Unknown data_type %s' % data_type)
    if data_type == 'train':
        return True
    elif data_type == 'test':
        return False


def get_network(name, data_type, images, num_classes):
    """ get specified network """
    is_training = check_network(name, data_type)
    net = networks_map[name]
    with arg_scope(net.arg_scope()):
        return net.model(images, num_classes, is_training)
