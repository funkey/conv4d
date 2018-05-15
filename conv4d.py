# -*- coding: UTF-8 -*-
from __future__ import print_function
import tensorflow as tf

def conv4d(
        input,
        filters,
        kernel_size,
        strides=(1, 1, 1, 1),
        padding='valid',
        data_format='channels_last',
        dilation_rate=(1, 1, 1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer=None,
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        trainable=True,
        name=None,
        reuse=None):
    '''Performs a 4D convolution (i.e., including the channel dimension) of a
    tensor ``(b, c, d, h, w)`` with ``k`` kernels. The output tensor will be of
    shape ``(b, c'*k, d', h', w')``. ``(c', d', h', w')`` will be smaller than
    ``(c, d, h, w)`` if a ``valid`` padding was chosen.

    This operator realizes a 4D convolution by performing several 3D
    convolutions. The following example demonstrates how this works for a 2D
    convolution as a sequence of 1D convolutions::

        I.shape == (h, w)
        k.shape == (U, V) and U%2 = V%2 = 1

        # we assume kernel is indexed as follows:
        u in [-U/2,...,U/2]
        v in [-V/2,...,V/2]

        (k*I)[i,j] = Σ_u Σ_v k[u,v] I[i+u,j+v]
                   = Σ_u (k[u]*I[i+u])[j]
        (k*I)[i]   = Σ_u k[u]*I[i+u]
        (k*I)      = Σ_u k[u]*I_u, with I_u[i] = I[i+u] shifted I by u

        Example:

            I = [
                [0,0,0],
                [1,1,1],
                [1,1,0],
                [1,0,0],
                [0,0,1]
            ]

            k = [
                [1,1,1],
                [1,2,1],
                [1,1,3],
            ]

            # convolve every row in I with every row in k, comments show output
            # row the convolution contributes to
            (I*k[0]) = [
                [0,0,0], # I[0] with k[0] ⇒ (k*I)[ 1] ✔
                [2,3,2], # I[1] with k[0] ⇒ (k*I)[ 2] ✔
                [2,2,1]  # I[2] with k[0] ⇒ (k*I)[ 3] ✔
                [1,1,0]  # I[3] with k[0] ⇒ (k*I)[ 4] ✔
                [0,1,1]  # I[4] with k[0] ⇒ (k*I)[ 5]
            ]
            (I*k[1]) = [
                [0,0,0], # I[0] with k[1] ⇒ (k*I)[ 0] ✔
                [3,4,3], # I[1] with k[1] ⇒ (k*I)[ 1] ✔
                [3,3,1]  # I[2] with k[1] ⇒ (k*I)[ 2] ✔
                [2,1,0]  # I[3] with k[1] ⇒ (k*I)[ 3] ✔
                [0,1,2]  # I[4] with k[1] ⇒ (k*I)[ 4] ✔
            ]
            (I*k[2]) = [
                [0,0,0], # I[0] with k[2] ⇒ (k*I)[-1]
                [4,5,2], # I[1] with k[2] ⇒ (k*I)[ 0] ✔
                [4,2,1]  # I[2] with k[2] ⇒ (k*I)[ 1] ✔
                [1,1,0]  # I[3] with k[2] ⇒ (k*I)[ 2] ✔
                [0,3,1]  # I[4] with k[2] ⇒ (k*I)[ 3] ✔
            ]

            # sum the contributions of all valid output rows (row 2 here)
            (k*I)[2] = (
                [2,3,2] +
                [3,3,1] +
                [1,1,0] +
            ) = [6,7,3]
    '''

    # check arguments
    assert len(input.get_shape().as_list()) == 5, (
        "Tensor of shape (b, c, d, h, w) expected")
    assert len(kernel_size) == 4, "4D kernel size expected"
    assert strides == (1, 1, 1, 1), (
        "Strides other than 1 not yet implemented")
    assert padding == 'valid', (
        "Padding other than 'valid' not yet implemented")
    assert data_format == 'channels_last', (
        "Data format other than 'channels_last' not yet implemented")
    assert dilation_rate == (1, 1, 1, 1), (
        "Dilation rate other than 1 not yet implemented")

    if not name:
        name = 'conv4d'

    # input, kernel, and output sizes
    (b, c_i, d_i, h_i, w_i) = tuple(input.get_shape().as_list())
    (c_k, d_k, h_k, w_k) = kernel_size

    # output size for 'valid' convolution
    (c_o, d_o, h_o, w_o) = (
        c_i - c_k + 1,
        d_i - d_k + 1,
        h_i - h_k + 1,
        w_i - w_k + 1
    )

    print("Input shape : ", (b, c_i, d_i, h_i, w_i))
    print("Kernel shape: ", (c_k, d_k, h_k, w_k))
    print("Output shape: ", (b, c_o*filters, d_o, h_o, w_o))

    # convolve each kernel channel i with each input channel j
    channel_results = [ None ]*c_o
    for i in range(c_k):

        # reuse variables of previous 3D convolutions for the same kernel
        # channel (or if the user indicated to have all variables reused)
        reuse_kernel = reuse

        for j in range(c_i):

            # add results to this output channel
            out_channel = j - (i - c_k/2) - (c_i - c_o)/2
            # print("Conv of kernel %d with input %d goes to output %d"%(i, j, out_channel))
            if out_channel < 0 or out_channel >= c_o:
                # print("not a valid output channel, skipping")
                continue

            # convolve input channel j with kernel channel i
            channel_conv3d = tf.layers.conv3d(
                tf.reshape(input[:,j,:], (b, 1, d_i, h_i, w_i)),
                filters,
                kernel_size=(d_k, h_k, w_k),
                padding=padding,
                data_format='channels_first',
                activation=None,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                trainable=trainable,
                name=name + '_3dchan%d'%i,
                reuse=reuse_kernel)
            reuse_kernel = True

            # print("\tInput :", (b, 1, d_i, h_i, w_i))
            # print("\tKernel:", (d_k, h_k, w_k))
            # print("\tOutput:", channel_conv3d.get_shape().as_list())

            if channel_results[out_channel] is None:
                channel_results[out_channel] = channel_conv3d
            else:
                channel_results[out_channel] += channel_conv3d

    output = tf.concat(channel_results, axis=1)
    print("Output: ", output.get_shape().as_list())

    if activation:
        output = activation(output)

    return output

if __name__ == "__main__":

    import numpy as np

    i = np.round(np.random.random((1, 10, 10, 10, 10))*100)
    input = tf.constant(i, dtype=tf.float32)
    bias_init = tf.constant_initializer(0)

    output = conv4d(
        input,
        1,
        (3, 3, 3, 3),
        bias_initializer=bias_init)

    with tf.Session() as s:

        s.run(tf.global_variables_initializer())
        o = s.run(output)

        print("conv4d at (0, 0, 0, 0): ", o[0,0,0,0,0])
        i0 = i[0,0,0:3,0:3,0:3].flatten()
        i1 = i[0,1,0:3,0:3,0:3].flatten()
        i2 = i[0,2,0:3,0:3,0:3].flatten()

        k0 = tf.get_default_graph().get_tensor_by_name(
            'conv4d_3dchan0/kernel:0').eval().flatten()
        k1 = tf.get_default_graph().get_tensor_by_name(
            'conv4d_3dchan1/kernel:0').eval().flatten()
        k2 = tf.get_default_graph().get_tensor_by_name(
            'conv4d_3dchan2/kernel:0').eval().flatten()

        compare = (i0*k0 + i1*k1 + i2*k2).sum()
        print("manually computed value at (0, 0, 0, 0): ", compare)
