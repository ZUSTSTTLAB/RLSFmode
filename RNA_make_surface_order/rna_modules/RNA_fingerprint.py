import tensorflow as tf

import numpy as np

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class RNA_fingerprint:

    """
    The neural network model.
    """

    def count_number_parameters(self):
        total_parameters = 0
        for variable in tf.compat.v1.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            print(variable)
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            print(variable_parameters)
            total_parameters += variable_parameters
        print("Total number parameters: %d" % total_parameters)

    def frobenius_norm(self, tensor):
        square_tensor = tf.square(tensor)
        tensor_sum = tf.reduce_sum(square_tensor)
        frobenius_norm = tf.sqrt(tensor_sum)
        return frobenius_norm

    def build_sparse_matrix_softmax(self, idx_non_zero_values, X, dense_shape_A):
        A = tf.SparseTensorValue(idx_non_zero_values, tf.squeeze(X), dense_shape_A)
        A = tf.sparse_reorder(A)  # n_edges x n_edges
        A = tf.sparse_softmax(A)

        return A

    def compute_initial_coordinates(self):
        range_rho = [0.0, self.max_rho]
        range_theta = [0, 2 * np.pi]

        grid_rho = np.linspace(range_rho[0], range_rho[1], num=self.n_rhos + 1)
        grid_rho = grid_rho[1:]
        grid_theta = np.linspace(range_theta[0], range_theta[1], num=self.n_thetas + 1)
        grid_theta = grid_theta[:-1]

        grid_rho_, grid_theta_ = np.meshgrid(grid_rho, grid_theta, sparse=False)  

        grid_rho_ = (
            grid_rho_.T
        ) 
        grid_theta_ = (
            grid_theta_.T
        ) 
        grid_rho_ = grid_rho_.flatten()
        grid_theta_ = grid_theta_.flatten()

        coords = np.concatenate((grid_rho_[None, :], grid_theta_[None, :]), axis=0)
        coords = coords.T  # every row contains the coordinates of a grid intersection

        return coords

    def inference(
        self,
        input_feat,
        rho_coords,
        theta_coords,
        mask,
        W_conv,
        b_conv,
        mu_rho,
        sigma_rho,
        mu_theta,
        sigma_theta,
        eps=1e-5,
        mean_gauss_activation=True,
    ):
        n_samples = tf.shape(rho_coords)[0]
        n_vertices = tf.shape(rho_coords)[1]
        n_feat = tf.shape(input_feat)[2]

        all_conv_feat = []
        for k in range(self.n_rotations):
            rho_coords_ = tf.reshape(rho_coords, [-1, 1])  # batch_size*n_vertices
            thetas_coords_ = tf.reshape(theta_coords, [-1, 1])  # batch_size*n_vertices

            thetas_coords_ += k * 2 * np.pi / self.n_rotations
            thetas_coords_ = tf.math.mod(thetas_coords_, 2 * np.pi)
            rho_coords_ = tf.exp(
                -tf.square(rho_coords_ - mu_rho) / (tf.square(sigma_rho) + eps)
            )
            thetas_coords_ = tf.exp(
                -tf.square(thetas_coords_ - mu_theta) / (tf.square(sigma_theta) + eps)
            )
            self.myshape = tf.shape(rho_coords_)
            self.rho_coords_debug = rho_coords_
            self.thetas_coords_debug = thetas_coords_

            gauss_activations = tf.multiply(
                rho_coords_, thetas_coords_
            )  # batch_size*n_vertices, n_gauss
            gauss_activations = tf.reshape(
                gauss_activations, [n_samples, n_vertices, -1]
            )  # batch_size, n_vertices, n_gauss
            gauss_activations = tf.multiply(gauss_activations, mask)
            if (
                mean_gauss_activation
            ):  # computes mean weights for the different gaussians
                gauss_activations /= (
                    tf.reduce_sum(gauss_activations, 1, keepdims=True) + eps
                )  # batch_size, n_vertices, n_gauss

            gauss_activations = tf.expand_dims(
                gauss_activations, 2
            )  # batch_size, n_vertices, 1, n_gauss,
            input_feat_ = tf.expand_dims(
                input_feat, 3
            )  # batch_size, n_vertices, n_feat, 1

            gauss_desc = tf.multiply(
                gauss_activations, input_feat_
            )  # batch_size, n_vertices, n_feat, n_gauss,
            gauss_desc = tf.reduce_sum(gauss_desc, 1)  # batch_size, n_feat, n_gauss,
            gauss_desc = tf.reshape(
                gauss_desc, [n_samples, self.n_thetas * self.n_rhos * n_feat]
            )  # batch_size, self.n_thetas*self.n_rhos*n_feat

            conv_feat = tf.matmul(gauss_desc, W_conv) + b_conv  # batch_size, n_gauss
            all_conv_feat.append(conv_feat)
        all_conv_feat = tf.stack(all_conv_feat)
        conv_feat = tf.reduce_max(all_conv_feat, 0)
        conv_feat = tf.nn.relu(conv_feat)
        return conv_feat

    def compute_data_loss(self, neg_thresh=1e1):
        pos_thresh = 4.0
        neg_thresh = 0.0
        pos_labels = self.labels[:, 0]
        n_pos = tf.reduce_sum(pos_labels)
        pos_scores = tf.multiply(self.logits[:, 0], tf.to_float(pos_labels))
        pos_scores = tf.reduce_sum(pos_scores) / n_pos

        neg_labels = self.labels[:, 1]
        n_neg = tf.reduce_sum(neg_labels)
        neg_scores = tf.multiply(self.logits[:, 1], tf.to_float(neg_labels))
        neg_scores = tf.reduce_sum(neg_scores) / n_neg

        data_loss = neg_scores - pos_scores
        return data_loss

    def __init__(
        self,
        max_rho,
        n_thetas=16,
        n_rhos=5,
        n_gamma=1.0,
        learning_rate=1e-3,
        n_rotations=16,
        idx_gpu="/device:CPU:0",
        feat_mask=[1.0, 1.0, 1.0, 1.0, 1.0],
        n_conv_layers=1,
        optimizer_method="Adam",
    ):

        # order of the spectral filters
        self.max_rho = max_rho
        self.n_thetas = n_thetas
        self.n_rhos = n_rhos

        self.sigma_rho_init = (
            max_rho / 8
        )  # in MoNet was 0.005 with max radius=0.04 (i.e. 8 times smaller)
        self.sigma_theta_init = 1.0  # 0.25
        self.n_rotations = n_rotations
        self.n_feat = int(sum(feat_mask))
        self.n_labels = 2


        with tf.Graph().as_default() as g:
            self.graph = g
            # tf.set_random_seed(0)
            # tf.random.set_seed(0)
            tf.compat.v1.set_random_seed(0)
            with tf.device(idx_gpu):

                initial_coords = self.compute_initial_coordinates()
                mu_rho_initial = np.expand_dims(initial_coords[:, 0], 0).astype(
                    "float32"
                )#rho旋转长度3,6,9
                mu_theta_initial = np.expand_dims(initial_coords[:, 1], 0).astype(
                    "float32"
                )#theta旋转角度
                self.mu_rho = []
                self.mu_theta = []
                self.sigma_rho = []
                self.sigma_theta = []
                for i in range(self.n_feat):
                    self.mu_rho.append(
                        tf.Variable(mu_rho_initial, name="mu_rho_{}".format(i))
                    )  # 1, n_gauss
                    self.mu_theta.append(
                        tf.Variable(mu_theta_initial, name="mu_theta_{}".format(i))
                    )  # 1, n_gauss
                    self.sigma_rho.append(
                        tf.Variable(
                            np.ones_like(mu_rho_initial) * self.sigma_rho_init,
                            name="sigma_rho_{}".format(i),
                        )
                    )  # 1, n_gauss   sigma_rho值为均为sigma_rho_init=9/8=1.25
                    self.sigma_theta.append(
                        tf.Variable(
                            (np.ones_like(mu_theta_initial) * self.sigma_theta_init),
                            name="sigma_theta_{}".format(i),
                        )
                    )  # 1, n_gauss  

                if n_conv_layers > 1:
                    self.mu_rho_l2 = tf.Variable(
                        mu_rho_initial, name="mu_rho_{}".format("l2")
                    )
                    self.sigma_rho_l2 = tf.Variable(
                        mu_theta_initial, name="mu_theta_{}".format("l2")
                    )
                    self.mu_theta_l2 = tf.Variable(
                        np.ones_like(mu_rho_initial) * self.sigma_rho_init,
                        name="sigma_rho_{}".format("l2"),
                    )
                    self.sigma_theta_l2 = tf.Variable(
                        (np.ones_like(mu_theta_initial) * self.sigma_theta_init),
                        name="sigma_theta_{}".format("l2"),
                    )
                if n_conv_layers > 2:
                    self.mu_rho_l3 = tf.Variable(
                        mu_rho_initial, name="mu_rho_{}".format("l3")
                    )
                    self.sigma_rho_l3 = tf.Variable(
                        mu_theta_initial, name="mu_theta_{}".format("l3")
                    )
                    self.mu_theta_l3 = tf.Variable(
                        np.ones_like(mu_rho_initial) * self.sigma_rho_init,
                        name="sigma_rho_{}".format("l3"),
                    )
                    self.sigma_theta_l3 = tf.Variable(
                        (np.ones_like(mu_theta_initial) * self.sigma_theta_init),
                        name="sigma_theta_{}".format("l3"),
                    )
                if n_conv_layers > 3:
                    self.mu_rho_l4 = tf.Variable(
                        mu_rho_initial, name="mu_rho_{}".format("l4")
                    )
                    self.sigma_rho_l4 = tf.Variable(
                        mu_theta_initial, name="mu_theta_{}".format("l4")
                    )
                    self.mu_theta_l4 = tf.Variable(
                        np.ones_like(mu_rho_initial) * self.sigma_rho_init,
                        name="sigma_rho_{}".format("l4"),
                    )
                    self.sigma_theta_l4 = tf.Variable(
                        (np.ones_like(mu_theta_initial) * self.sigma_theta_init),
                        name="sigma_theta_{}".format("l4"),
                    )
                # tf.placeholder
                self.rho_coords = tf.compat.v1.placeholder(
                    tf.float32
                )  # batch_size, n_vertices, 1
                self.theta_coords = tf.compat.v1.placeholder(
                    tf.float32
                )  # batch_size, n_vertices, 1
                self.input_feat = tf.compat.v1.placeholder(
                    tf.float32, shape=[None, None, self.n_feat]
                )  # batch_size, n_vertices, n_feat
                self.mask = tf.compat.v1.placeholder(tf.float32)  # batch_size, n_vertices, 1

                self.pos_idx = tf.compat.v1.placeholder(tf.int32)  # batch_size/2
                self.neg_idx = tf.compat.v1.placeholder(tf.int32)  # batch_size/2
                self.labels = tf.compat.v1.placeholder(tf.int32)  # batch_size, n_labels
                self.indices_tensor = tf.compat.v1.placeholder(
                    tf.int32
                )  # batch_size, max_verts (< 30)
                self.keep_prob = tf.compat.v1.placeholder(tf.float32)  # scalar

                self.global_desc = []

                # Use Geometric deep learning
                b_conv = []
                for i in range(self.n_feat):
                    b_conv.append(
                        tf.Variable(
                            tf.zeros([self.n_thetas * self.n_rhos]),
                            name="b_conv_{}".format(i),
                        )
                    )
                for i in range(self.n_feat):
                    my_input_feat = tf.expand_dims(self.input_feat[:, :, i], 2)
                    # tf.get_variable
                    W_conv = tf.compat.v1.get_variable(
                        "W_conv_{}".format(i),
                        shape=[
                            self.n_thetas * self.n_rhos,
                            self.n_thetas * self.n_rhos,
                        ],
                        initializer=tf.contrib.layers.xavier_initializer(),
                    )

                    rho_coords = self.rho_coords
                    theta_coords = self.theta_coords
                    mask = self.mask

                    self.global_desc.append(
                        self.inference(
                            my_input_feat,
                            rho_coords,
                            theta_coords,
                            mask,
                            W_conv,
                            b_conv[i],
                            self.mu_rho[i],
                            self.sigma_rho[i],
                            self.mu_theta[i],
                            self.sigma_theta[i],
                        )
                    )  # batch_size, n_gauss*1
                # global_desc is n_feat, batch_size, n_gauss*1
                # They should be batch_size, n_feat*n_gauss (5 x 12)

                self.global_desc = tf.stack(self.global_desc, axis=1)

                self.global_desc = tf.reshape(
                    self.global_desc, [-1, self.n_thetas * self.n_rhos * self.n_feat]
                )

                self.global_desc_copy = self.global_desc  
                # Create a session for running Ops on the Graph.
                config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
                config.gpu_options.allow_growth = True
                self.session = tf.compat.v1.Session(config=config)

                # Run the Op to initialize the variables.
                init = tf.compat.v1.global_variables_initializer()
                self.session.run(init)
                self.count_number_parameters()


