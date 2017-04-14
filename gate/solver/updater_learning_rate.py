import tensorflow as tf


def configure(dataset, num_samples_per_epoch, global_step):
    """Configures the learning rate.

    Args:
      num_samples_per_epoch: The number of samples in each epoch of training.
      global_step: The global_step tensor.

    Returns:
      A `Tensor` representing the learning rate.

    Raises:
      ValueError: if
    """
    current_base_lr = dataset.lr.learning_rate

    decay_steps = int(num_samples_per_epoch / dataset.batch_size *
                      dataset.lr.num_epochs_per_decay)

    if dataset.lr.sync_replicas:
        decay_steps /= dataset.lr.replicas_to_aggregate

    if dataset.lr.learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(current_base_lr,
                                          global_step,
                                          decay_steps,
                                          dataset.lr.learning_rate_decay_factor,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')

    elif dataset.lr.learning_rate_decay_type == 'fixed':
        return tf.constant(current_base_lr, name='fixed_learning_rate')

    elif dataset.lr.learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(current_base_lr,
                                         global_step,
                                         decay_steps,
                                         dataset.lr.end_learning_rate,
                                         power=1.0,
                                         cycle=False,
                                         name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized',
                         dataset.lr.learning_rate_decay_type)
