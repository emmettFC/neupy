import numpy as np

from neupy import layers
from neupy.utils import asfloat

from base import BaseTestCase


class EmbeddingLayerTestCase(BaseTestCase):
    def test_embedding_layer(self):
        weight = np.arange(10).reshape((5, 2))

        network = layers.join(
            layers.Input(1),
            layers.Embedding(5, 2, weight=weight),
        )

        input_vector = asfloat(np.array([[0, 1, 4]]).T)
        expected_output = np.array([
            [[0, 1]],
            [[2, 3]],
            [[8, 9]],
        ])
        actual_output = self.eval(network.output(input_vector))

        self.assertShapesEqual(network.output_shape, (None, 1, 2))
        np.testing.assert_array_equal(expected_output, actual_output)

    def test_embedding_layer_repr(self):
        self.assertEqual(
            str(layers.Embedding(5, 2)),
            (
                "Embedding(5, 2, weight=HeNormal(gain=1.0), "
                "name='embedding-1')"
            )
        )

    def test_embedding_variables(self):
        network = layers.join(
            layers.Input(2),
            layers.Embedding(3, 5, name='embed'),
        )
        self.assertDictEqual(network.layer('embed').variables, {})

        network.outputs
        variables = network.layer('embed').variables
        self.assertSequenceEqual(list(variables.keys()), ['weight'])
        self.assertShapesEqual(variables['weight'].shape, (3, 5))
