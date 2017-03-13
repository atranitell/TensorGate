
from nets import net_cifarnet


def get_network(name, data_type, images, num_classes):
    if data_type == 'train':
        is_training = True
    elif data_type == 'test':
        is_training = False
    else:
        raise ValueError('Unknown data_type %s' % data_type)

    if name is 'cifarnet':
        return net_cifarnet.cifarnet().model(images, num_classes, is_training)
    else:
        raise ValueError('Unknown model name %s' % name)
