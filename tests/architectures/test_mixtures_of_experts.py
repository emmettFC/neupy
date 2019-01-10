import copy

import numpy as np
from sklearn import datasets, preprocessing, model_selection

from neupy.utils import asfloat
from neupy.exceptions import LayerConnectionError
from neupy.algorithms.gd import objectives
from neupy import algorithms, layers, architectures

from base import BaseTestCase


class MixtureOfExpertsTestCase(BaseTestCase):
    def setUp(self):
        super(MixtureOfExpertsTestCase, self).setUp()
        self.networks = [
            algorithms.GradientDescent(
                network=[
                    layers.Input(1),
                    layers.Sigmoid(20),
                    layers.Sigmoid(1)
                ],
                step=0.2,
                batch_size=None,
                verbose=False
            ),
            layers.join(
                layers.Input(1),
                layers.Sigmoid(20),
                layers.Sigmoid(1)
            ),
        ]

    def test_mixture_of_experts_problem_with_specific_network(self):
        with self.assertRaisesRegexp(ValueError, "specified as a list"):
            architectures.mixture_of_experts(*self.networks)

        with self.assertRaisesRegexp(ValueError, "has more than one input"):
            last_network = layers.join(
                layers.parallel(
                    layers.Input(1),
                    layers.Input(2),
                ),
                layers.Concatenate(),
            )
            architectures.mixture_of_experts(
                instances=self.networks + [last_network])

        with self.assertRaisesRegexp(ValueError, "has more than one output"):
            last_network = layers.join(
                layers.Input(1),
                layers.parallel(
                    layers.Softmax(1),
                    layers.Softmax(1),
                ),
            )
            architectures.mixture_of_experts(
                instances=self.networks + [last_network])

        error_message = (
            "Each network from the mixture of experts has to "
            "process only 2-dimensional inputs. Network #2.+"
            "Input layer's shape: \(\?, 1, 1, 1\)"
        )
        with self.assertRaisesRegexp(ValueError, error_message):
            last_network = layers.Input((1, 1, 1))
            architectures.mixture_of_experts(
                instances=self.networks + [last_network])

    def test_mixture_of_experts_problem_with_incompatible_networks(self):
        error_message = "Networks have incompatible input shapes."
        with self.assertRaisesRegexp(ValueError, error_message):
            architectures.mixture_of_experts(
                instances=self.networks + [layers.Input(10)])

        error_message = "Networks have incompatible output shapes."
        with self.assertRaisesRegexp(ValueError, error_message):
            architectures.mixture_of_experts(
                instances=self.networks + [
                    layers.Input(1) >> layers.Relu(10)
                ])

    def test_mixture_of_experts_init_gating_network_exceptions(self):
        with self.assertRaisesRegexp(ValueError, "Invalid type"):
            architectures.mixture_of_experts(
                instances=self.networks,
                gating_layer=(layers.Input(1) >> layers.Softmax(2)))

        error_message = (
            "Gating layer can work only for combining only "
            "10 networks, got 2 networks instead."
        )
        with self.assertRaisesRegexp(LayerConnectionError, error_message):
            architectures.mixture_of_experts(
                instances=self.networks,
                gating_layer=layers.Softmax(10))

    def test_mixture_of_experts_multi_class_classification(self):
        insize, outsize = (10, 3)
        n_epochs = 10

        default_configs = dict(
            step=0.1,
            batch_size=10,
            loss='categorical_crossentropy',
            verbose=False)

        architecture = layers.join(
            layers.Input(insize),
            layers.Relu(20),
            layers.Softmax(outsize))

        data, target = datasets.make_classification(
            n_samples=200,
            n_features=insize,
            n_classes=outsize,
            n_clusters_per_class=2,
            n_informative=5)

        input_scaler = preprocessing.MinMaxScaler((-1, 1))
        one_hot = preprocessing.OneHotEncoder(categories='auto')

        target = target.reshape((-1, 1))
        encoded_target = one_hot.fit_transform(target)
        x_train, x_test, y_train, y_test = model_selection.train_test_split(
            input_scaler.fit_transform(data),
            np.asarray(encoded_target.todense()),
            test_size=0.2)

        # -------------- Train single GradientDescent -------------- #

        bpnet = algorithms.Momentum(
            copy.deepcopy(architecture),
            **default_configs
        )

        bpnet.train(x_train, y_train, epochs=n_epochs)
        network_output = bpnet.predict(x_test)

        network_error = self.eval(
            objectives.categorical_crossentropy(y_test, network_output))

        # -------------- Train ensemlbe -------------- #

        moe = algorithms.Momentum(
            architectures.mixture_of_experts([
                copy.deepcopy(architecture),
                copy.deepcopy(architecture),
                copy.deepcopy(architecture),
            ]),
            **default_configs
        )
        moe.train(x_train, y_train, epochs=n_epochs)
        ensemble_output = moe.predict(x_test)

        ensemlbe_error = self.eval(
            objectives.categorical_crossentropy(y_test, ensemble_output))

        self.assertGreater(network_error, ensemlbe_error)

    def test_mixture_of_experts_architecture(self):
        network = architectures.mixture_of_experts([
            layers.join(
                layers.Input(10),
                layers.Relu(5),
            ),
            layers.join(
                layers.Input(10),
                layers.Relu(20),
                layers.Relu(5),
            ),
            layers.join(
                layers.Input(10),
                layers.Relu(30),
                layers.Relu(40),
                layers.Relu(5),
            ),
        ])

        self.assertEqual(len(network), 12)
        self.assertShapesEqual(network.input_shape, (None, 10))
        self.assertShapesEqual(network.output_shape, (None, 5))

        random_input = asfloat(np.random.random((3, 10)))
        prediction = self.eval(network.output(random_input))

        self.assertEqual(prediction.shape, (3, 5))
