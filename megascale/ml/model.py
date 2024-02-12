import flax.linen as nn
import jax


class OneDimensionalCNNModel(nn.Module):
    """Implementation of the 1D-CNN model."""

    conv_features: int
    dense_features: int
    without_pooling: bool

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """Call function of the model."""
        # Define layers
        conv1 = nn.Conv(features=self.conv_features, kernel_size=(3,), padding="SAME")
        conv2 = nn.Conv(features=self.conv_features, kernel_size=(3,), padding="SAME")
        dense1 = nn.Dense(self.dense_features)
        dense2 = nn.Dense(self.dense_features // 10)

        # Apply conv1
        x = conv1(x)
        # Apply ReLU
        x = nn.relu(x)
        # Apply conv2
        x = conv2(x)
        # Apply ReLU
        x = nn.relu(x)

        # If requested, apply avg. pooling
        if not self.without_pooling:
            x = nn.avg_pool(x, window_shape=(3,), padding="VALID")

        # Flatten
        x = x.reshape((x.shape[0], -1))

        # Apply dense1
        x = dense1(x)
        # Apply ReLU
        x = nn.relu(x)
        # Apply dense1
        x = dense2(x)
        # Apply ReLU
        x = nn.relu(x)

        x = nn.Dense(1)(x)
        return x
