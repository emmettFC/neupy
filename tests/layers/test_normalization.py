import unittest
import tempfile

import numpy as np
import tensorflow as tf
from scipy import stats

from neupy import layers, algorithms, storage
from neupy.utils import asfloat
from neupy.exceptions import LayerConnectionError

from base import BaseTestCase
from helpers import simple_classification


class BatchNormTestCase(BaseTestCase):
    def test_batch_norm_as_shared_variable(self):
        gamma = tf.Variable(
            asfloat(np.ones((1, 2))),
            name='gamma',
            dtype=tf.float32,
        )
        beta = tf.Variable(
            asfloat(2 * np.ones((1, 2))),
            name='beta',
            dtype=tf.float32,
        )

        batch_norm = layers.BatchNorm(gamma=gamma, beta=beta)
        network = layers.join(layers.Input(2), batch_norm)
        network.outputs

        self.assertIs(gamma, batch_norm.gamma)
        self.assertIs(beta, batch_norm.beta)

    def test_simple_batch_norm(self):
        network = layers.Input(10) > layers.BatchNorm()

        input_value = tf.Variable(
            asfloat(np.random.random((30, 10))),
            name='input_value',
            dtype=tf.float32,
        )
        output_value = self.eval(network.output(input_value, training=True))

        self.assertTrue(stats.mstats.normaltest(output_value))
        self.assertAlmostEqual(output_value.mean(), 0, places=3)
        self.assertAlmostEqual(output_value.std(), 1, places=3)

    def test_batch_norm_gamma_beta_params(self):
        default_beta = -3.14
        default_gamma = 4.3
        network = layers.join(
            layers.Input(10),
            layers.BatchNorm(gamma=default_gamma, beta=default_beta)
        )

        input_value = tf.Variable(
            asfloat(np.random.random((30, 10))),
            name='input_value',
            dtype=tf.float32,
        )
        output_value = self.eval(network.output(input_value, training=True))

        self.assertAlmostEqual(output_value.mean(), default_beta, places=3)
        self.assertAlmostEqual(output_value.std(), default_gamma, places=3)

    def test_batch_norm_between_layers(self):
        network = layers.join(
            layers.Input(10),
            layers.Relu(40),
            layers.BatchNorm(),
            layers.Relu(1),
        )

        input_value = tf.Variable(
            asfloat(np.random.random((30, 10))),
            name='input_value',
            dtype=tf.float32,
        )
        outpu_value = self.eval(network.output(input_value, training=True))
        self.assertEqual(outpu_value.shape, (30, 1))

    def test_batch_norm_in_non_training_state(self):
        batch_norm = layers.BatchNorm()
        layers.Input(10) > batch_norm

        input_value = tf.Variable(
            asfloat(np.random.random((30, 10))),
            name='input_value',
            dtype=tf.float32,
        )

        self.assertEqual(len(batch_norm.updates), 0)

        batch_norm.output(input_value, training=True)
        self.assertEqual(len(batch_norm.updates), 2)

        # Without training your running mean and std suppose to be
        # equal to 0 and 1 respectavely.
        output_value = self.eval(batch_norm.output(input_value))
        np.testing.assert_array_almost_equal(
            self.eval(input_value), output_value, decimal=4)

    @unittest.skip("Storage tests are broken")
    def test_batch_norm_storage(self):
        x_train, x_test, y_train, y_test = simple_classification()

        batch_norm = layers.BatchNorm()
        gdnet = algorithms.GradientDescent(
            [
                layers.Input(10),
                layers.Relu(5),
                batch_norm,
                layers.Sigmoid(1),
            ],
            batch_size=10,
            verbose=True,  # keep it as `True`
        )
        gdnet.train(x_train, y_train, epochs=5)

        error_before_save = gdnet.score(x_test, y_test)
        mean_before_save = self.eval(batch_norm.running_mean)
        variance_before_save = self.eval(batch_norm.running_inv_std)

        with tempfile.NamedTemporaryFile() as temp:
            storage.save(gdnet, temp.name)
            storage.load(gdnet, temp.name)

            error_after_load = gdnet.score(x_test, y_test)
            mean_after_load = self.eval(batch_norm.running_mean)
            variance_after_load = self.eval(batch_norm.running_inv_std)

            self.assertAlmostEqual(error_before_save, error_after_load)
            np.testing.assert_array_almost_equal(
                mean_before_save, mean_after_load)

            np.testing.assert_array_almost_equal(
                variance_before_save, variance_after_load)


class LocalResponseNormTestCase(BaseTestCase):
    def test_local_response_norm_exceptions(self):
        with self.assertRaisesRegexp(ValueError, "Only works with odd"):
            layers.LocalResponseNorm(depth_radius=2)

        with self.assertRaises(LayerConnectionError):
            layers.join(layers.Input(10), layers.LocalResponseNorm())

    def test_local_response_normalization_layer(self):
        network = layers.join(
            layers.Input((1, 1, 1)),
            layers.LocalResponseNorm(),
        )

        x_tensor = asfloat(np.ones((1, 1, 1, 1)))
        actual_output = self.eval(network.output(x_tensor, training=True))
        expected_output = np.array([0.59458]).reshape((-1, 1, 1, 1))

        np.testing.assert_array_almost_equal(
            expected_output, actual_output, decimal=5)
