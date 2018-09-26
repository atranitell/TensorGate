import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import abc


class CustomOp():

  def __init__(self, inp, Tout, stateful=False, name=None, graph=None):
    self.inp = inp
    self.Tout = Tout
    self.stateful = stateful
    self.name = name
    # graph
    self.graph = graph
    if self.graph is None:
      self.graph = tf.get_default_graph()

  @abc.abstractmethod
  def forward(self, *args):
    assert NotImplementedError

  @abc.abstractmethod
  def backward(self, op, grad):
    return self._internel_backward()

  def gradient(self, *args):
    assert NotImplementedError

  def _internel_forward(self):
    random_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
    tf.RegisterGradient(random_name)(self.backward)
    override_content = {"PyFunc": random_name, "PyFuncStateless": random_name}
    with self.graph.gradient_override_map(override_content):
      return tf.py_func(self.forward, self.inp, self.Tout,
                        stateful=self.stateful, name=self.name)

  def _internel_backward(self):
    with ops.name_scope(self.name, "OpGrad", self.inp) as name:
      return tf.py_func(self.gradient, self.inp, self.Tout,
                        name=name, stateful=False)

  def build(self):
    with ops.name_scope(self.name, "Op", self.inp):
      return self._internel_forward()[0]


class MyReLU(CustomOp):

  def __init__(self, inp, Tout, stateful=False, name='my_relu'):
    self.threshold = 0.05
    super().__init__(inp, Tout, stateful, name)

  def forward(self, x, x1):
    print('forward', x.shape, x1.shape)
    res_x = []
    for it in x:
      if it < self.threshold:
        res_x.append(0.0)
      else:
        res_x.append(it)
    return np.array(res_x).astype(np.float32)

  def gradient(self, x, x1):
    print('backward', x.shape, x1.shape)
    res_x = []
    for it in x:
      if it < self.threshold:
        res_x.append(0.0)
      else:
        res_x.append(1.0)
    return np.array(res_x).astype(np.float32)

  def backward(self, op, grad):
    cur_grad = super().backward(op, grad)
    print(cur)
    next_grad = grad * cur_grad
    return next_grad


with tf.Session() as sess:
  x_tf = tf.constant([-0.3, 0.005, 0.08, 0.12])
  x1_tf = tf.constant([-0.4, 0.0051, 0.22])
  y_tf = MyReLU([x_tf, x1_tf], [tf.float32]).build()
  # y = my_relu_tf(x)
  tf.global_variables_initializer().run()
  print(x_tf.eval())
  print(y_tf.eval())
  print(tf.gradients(y_tf, [x_tf, x1_tf])[0].eval())
