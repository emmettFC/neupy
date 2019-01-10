Debug layer connections
=======================

Explore connection shapes
--------------------------

The simplest way to debug network is just to explore input and output shapes.

.. code-block:: python

    >>> from neupy import layers
    >>> network = layers.Input(10) > layers.Relu(5) > layers.Softmax(3)
    >>> connection
    Input(10) > Relu(5) > Softmax(3)
    >>>
    >>> connection.input_shape
    (10,)
    >>> connection.output_shape
    (3,)

The ``input_shape`` and ``output_shape`` properties store information about expected shape of the network's input and it's output shape after propagation. Notice that shapes do not store information about batch size. In our example, we might assume that we want to propagate 7 samples and network expects input matrix with shape ``(7, 10)`` and for this input it will output matrix with shape ``(7, 3)``.

Every layer in the network also has information about input and output shapes. We can check each layer separately.

.. code-block:: python

    >>> for layer in connection:
    ...     print(layer)
    ...     print("Input shape: {}".format(layer.input_shape))
    ...     print("Output shape: {}".format(layer.output_shape))
    ...     print()
    ...
    Input(10)
    Input shape: (10,)
    Output shape: (10,)

    Relu(5)
    Input shape: (10,)
    Output shape: (5,)

    Softmax(3)
    Input shape: (5,)
    Output shape: (3,)

For the networks that have parallel connection NeuPy will topologically sort all layers first and then presented one by one during the iteration.

Visualize connections
---------------------

For the debugging, it's useful to explore network's architecture. It's possible to visualize network in the form of a graph. Let's say we have this network.

.. code-block:: python

    from neupy import layers

    network = layers.join(
        layers.Input((10, 10, 3)),

        [[
            layers.Convolution((3, 3, 32)),
            layers.Relu(),
            layers.MaxPooling((2, 2)),
        ], [
            layers.Convolution((7, 7, 16)),
            layers.Relu(),
        ]],
        layers.Concatenate(),

        layers.Reshape(),
        layers.Softmax(10),
    )

To be able to visualize it we can just use :class:`network_structure <neupy.plots.network_structure>` function.

.. code-block:: python

    from neupy import plots
    plots.network_structure(network)

.. raw:: html

    <br>

.. image:: images/layer-structure-debug.png
    :width: 90%
    :align: center
    :alt: Debug network structure

This function will pop-up PDF file with a graph that defines all layers and relations between them. In addition, it shows input and output shape per each layer.

Instead of showing pop-up preview we can simply save it in the separate file.

.. code-block:: python

    from neupy import plots
    plots.network_structure(
        connection,
        filepath='connection.pdf',
        show=False,
    )

Function also works for the training algorithms with constructible architectures. It just automatically extracts architecture from the algorithm and visualizes it.

.. code-block:: python

    from neupy import algorithms, plots

    nnet = algorithms.GradientDescent((2, 3, 1))
    plots.network_structure(nnet)

.. raw:: html

    <br>

.. image:: images/network-structure-debug.png
    :width: 60%
    :align: center
    :alt: Debug network structure

Count number of parameters
--------------------------

The ``count_parameters`` function allow to go through the network and count total number of parameters in it.

.. code-block:: python

    >>> from neupy import layers
    >>>
    >>> network = layers.join(
    ...     layers.Input(10),
    ...     layers.Relu(5),  # weight: 10 * 5, bias: 5, total: 55
    ...     layers.Relu(2),  # weight:  5 * 2, bias: 2, total: 12
    ... )
    >>> layers.count_parameters(connection)
    67

Iterate through all network parameters
--------------------------------------

.. code-block:: python

    >>> network = layers.join(
    ...     layers.Input(1),
    ...     layers.Sigmoid(2),
    ...     layers.Sigmoid(3),
    ... )
    >>>
    >>> print(network)
    Input(1) > Sigmoid(2) > Sigmoid(3)
    >>>
    >>> for (layer, varname), variable in network.variables.items():
    ...     print("Layer: {}".format(layer))
    ...     print("Name: {}".format(varname))
    ...     print("Variable: {}".format(variable))
    ...     print()
    ...
    Layer: Sigmoid(2)
    Name: weight
    Variable: <tf.Variable 'layer/sigmoid-3/weight:0' shape=(1, 2) dtype=float32_ref>

    Layer: Sigmoid(2)
    Name: bias
    Variable: <tf.Variable 'layer/sigmoid-3/bias:0' shape=(2,) dtype=float32_ref>

    Layer: Sigmoid(3)
    Name: weight
    Variable: <tf.Variable 'layer/sigmoid-4/weight:0' shape=(2, 3) dtype=float32_ref>

    Layer: Sigmoid(3)
    Name: bias
    Variable: <tf.Variable 'layer/sigmoid-4/bias:0' shape=(3,) dtype=float32_ref>

Exploring graph connections
---------------------------

Any relation between layers is stored in the graph. To be able to debug connections we can check network's graph to make sure that all connections defined correctly.

.. code-block:: python

    >>> from neupy import layers
    >>>
    >>> input_layer = layers.Input(10)
    >>> input_layer.graph
    [(Input(10), [])]

Since layer is not connected to any other layer the graph is empty. We can define network with more layers and check it's graph.

.. code-block:: python

    >>> network = layers.join(
    ...     input_layer,
    ...     [[
    ...         layers.Relu(10),
    ...         layers.Relu(20),
    ...     ], [
    ...         layers.Relu(30),
    ...     ]],
    ...     layers.Concatenate()
    ... )
    >>> network.graph
    [(Input(10), [Relu(10), Relu(30)]),
     (Relu(10), [Relu(20)]),
     (Relu(20), [Concatenate()]),
     (Relu(30), [Concatenate()]),
     (Concatenate(), [])]

The graph has formatted representation. If we need to access it directly then we should check the ``forward_graph`` attribute.

.. code-block:: python

    >>> network.graph.forward_graph
    OrderedDict([(Input(10), [Relu(10), Relu(30)]), (Relu(10),
    [Relu(20)]), (Relu(20), [Concatenate()]), (Relu(30),
    [Concatenate()]), (Concatenate(), [])])

**Do not try to modify graph**. Modifications can break relations between layers. This feature is only available for debugging.
