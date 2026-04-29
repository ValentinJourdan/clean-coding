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
        conv1 = nn.Conv(features=self.conv_features, kernel_size=(3,), padding="SAME")
        conv2 = nn.Conv(features=self.conv_features, kernel_size=(3,), padding="SAME")
        dense1 = nn.Dense(self.dense_features)
        dense2 = nn.Dense(self.dense_features // 10)

        x = conv1(x)
        x = nn.relu(x)
        x = conv2(x)
        x = nn.relu(x)

        if not self.without_pooling:
            x = nn.avg_pool(x, window_shape=(3,), padding="VALID")

        x = x.reshape((x.shape[0], -1))

        x = dense1(x)
        x = nn.relu(x)
        x = dense2(x)
        x = nn.relu(x)

        x = nn.Dense(1)(x)
        return x
