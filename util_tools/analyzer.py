"""Tools for analyzing the operations and variables in a TensorFlow graph."""


def desc_tensor(var):
    """Returns a compact and informative string about a tensor.

    Args:
      var: A tensor variable.

    Returns:
      a string with type and size, e.g.: (float32 1x8x8x1024).
    """
    description = '(name=' + str(var.name) + ' dtype=' + str(var.dtype.name)
    sizes = var.get_shape()
    description += ' shape=['
    for i, size in enumerate(sizes):
        description += str(size)
        if i < len(sizes) - 1:
            description += ','
    description += '])'
    return description


def desc_ops(graph, print_info=True):
    """Compute the estimated size of the ops.outputs in the graph.

    Args:
      graph: the graph containing the operations.
      print_info: Optional, if true print ops and their outputs.

    Returns:
      total size of the ops.outputs
    """
    if print_info:
        print('---------')
        print('Operations: name -> (type shapes) [size]')
        print('---------')
    total_size = 0
    for op in graph.get_operations():
        op_size = 0
        shapes = []
        for output in op.outputs:
            # if output.num_elements() is None or [] assume size 0.
            output_size = output.get_shape().num_elements() or 0
            if output.get_shape():
                shapes.append(desc_tensor(output))
            op_size += output_size
        if print_info:
            print(op.name, '\t->', ', '.join(shapes), '[' + str(op_size) + ']')
        total_size += op_size
    return total_size


def desc_variables(variables, print_info=True):
    """Prints the names and shapes of the variables.

    Args:
      variables: list of variables, for example tf.global_variables().
      print_info: Optional, if true print variables and their shape.

    Returns:
      (total size of the variables, total bytes of the variables)
    """
    if print_info:
        print('---------')
        print('Variables: name (type shape) [size]')
        print('---------')
    total_size = 0
    total_bytes = 0
    for var in variables:
        # if var.num_elements() is None or [] assume size 0.
        var_size = var.get_shape().num_elements() or 0
        var_bytes = var_size * var.dtype.size
        total_size += var_size
        total_bytes += var_bytes
        if print_info:
            print(var.name, desc_tensor(var), '[%d, bytes: %d]' %
                  (var_size, var_bytes))
    if print_info:
        print('Total size of variables: %d' % total_size)
        print('Total bytes of variables: %d' % total_bytes)
    return total_size, total_bytes
