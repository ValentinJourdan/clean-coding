import flax.linen as nn
import jax


class OneDimensionalCNNModel(nn.Module):
    """Implementation of the 1D-CNN model."""

    conv_features: int
    dense_features: int
    use_pooling: bool
    conv_kernel_size: int = 3
    pooling_window_size: int = 3
    dense_bottleneck_factor: int = 10
    output_size: int = 1

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """Call function of the model."""
        conv1 = nn.Conv(features=self.conv_features, kernel_size=(self.conv_kernel_size,), padding="SAME")
        conv2 = nn.Conv(features=self.conv_features, kernel_size=(self.conv_kernel_size,), padding="SAME")
        dense1 = nn.Dense(self.dense_features)
        dense2 = nn.Dense(self.dense_features // self.dense_bottleneck_factor)

        x = conv1(x)
        x = nn.relu(x)
        x = conv2(x)
        x = nn.relu(x)

        if not self.use_pooling:
            x = nn.avg_pool(x, window_shape=(self.pooling_window_size,), padding="VALID")

        x = x.reshape((x.shape[0], -1))

        x = dense1(x)
        x = nn.relu(x)
        x = dense2(x)
        x = nn.relu(x)

        x = nn.Dense(self.output_size)(x)
        return x
