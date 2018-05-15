conv4d
======

A [TensorFlow](https://www.tensorflow.org/) operator that realizes convolutions
in 4D by performing several 3D convolutions.

Performs a convolution of the ``(t, z, y, x)`` dimensions of a tensor with
shape ``(b, c, l, d, h, w)`` with ``k`` filters. The output tensor will be of
shape ``(b, k, l', d', h', w')``. ``(l', d', h', w')`` will be smaller than
``(l, d, h, w)`` if a ``valid`` padding was chosen.
