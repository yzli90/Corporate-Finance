import tensorflow as tf
import economics

"""
Neural Networks and Maliar (2021) Training Step.
"""

HIDDEN_DIM = 64
EULER_WEIGHT = 0.01  # Weight for Euler residuals in total loss


class ValueFunctionNet(tf.keras.Model):
    """Approximates V(k, z)"""

    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(HIDDEN_DIM, activation='relu')
        self.dense2 = tf.keras.layers.Dense(HIDDEN_DIM, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1, activation='linear')

    def call(self, inputs):
        # Scale k for stability
        k = inputs[:, 0:1] / 100.0
        z = inputs[:, 1:2]
        x = self.dense1(tf.concat([k, z], axis=1))
        x = self.dense2(x)
        return self.output_layer(x)


class PolicyNet(tf.keras.Model):
    """Approximates k' = Policy(k, z)"""

    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(HIDDEN_DIM, activation='relu')
        self.dense2 = tf.keras.layers.Dense(HIDDEN_DIM, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1, activation='softplus')  # Ensure k' > 0

    def call(self, inputs):
        k = inputs[:, 0:1] / 100.0
        z = inputs[:, 1:2]
        x = self.dense1(tf.concat([k, z], axis=1))
        x = self.dense2(x)
        return self.output_layer(x) * 100.0


@tf.function
def train_step_maliar_eq15(value_net, policy_net, optimizer, k_batch, z_batch):
    """
    Maliar (2021) Eq 15: Min E[Bellman_Resid^2 + nu * Euler_Resid^2]
    Uses All-in-One (AiO) integration with two uncorrelated shocks.
    """
    # Generate two independent shocks for AiO
    z_prime_1 = economics.transition_shock(z_batch, seed=1)
    z_prime_2 = economics.transition_shock(z_batch, seed=2)

    params = value_net.trainable_variables + policy_net.trainable_variables

    with tf.GradientTape() as tape:
        inputs = tf.stack([k_batch, z_batch], axis=1)
        k_prime = tf.squeeze(policy_net(inputs))
        v_curr = tf.squeeze(value_net(inputs))

        # --- 1. Bellman Residuals ---
        reward = economics.shareholder_cash_flow(k_batch, z_batch, k_prime)

        v_next_1 = tf.squeeze(value_net(tf.stack([k_prime, z_prime_1], axis=1)))
        v_next_2 = tf.squeeze(value_net(tf.stack([k_prime, z_prime_2], axis=1)))

        res_b1 = v_curr - (reward + economics.BETA * v_next_1)
        res_b2 = v_curr - (reward + economics.BETA * v_next_2)

        loss_bellman = tf.reduce_mean(res_b1 * res_b2)

        # --- 2. Euler Residuals (FOC) ---
        # Need gradients w.r.t k_prime
        with tf.GradientTape(persistent=True) as inner_tape:
            inner_tape.watch(k_prime)
            r_inner = economics.shareholder_cash_flow(k_batch, z_batch, k_prime)
            vn_1_in = tf.squeeze(value_net(tf.stack([k_prime, z_prime_1], axis=1)))
            vn_2_in = tf.squeeze(value_net(tf.stack([k_prime, z_prime_2], axis=1)))

        d_r_dk = inner_tape.gradient(r_inner, k_prime)
        d_v1_dk = inner_tape.gradient(vn_1_in, k_prime)
        d_v2_dk = inner_tape.gradient(vn_2_in, k_prime)
        del inner_tape

        # Clip gradients to handle steep fixed costs
        d_r_dk = tf.clip_by_value(d_r_dk, -20.0, 20.0)
        d_v1_dk = tf.clip_by_value(d_v1_dk, -20.0, 20.0)
        d_v2_dk = tf.clip_by_value(d_v2_dk, -20.0, 20.0)

        res_e1 = d_r_dk + economics.BETA * d_v1_dk
        res_e2 = d_r_dk + economics.BETA * d_v2_dk

        loss_euler = tf.reduce_mean(res_e1 * res_e2)

        total_loss = loss_bellman + EULER_WEIGHT * loss_euler

    # Update weights
    grads = tape.gradient(total_loss, params)
    grads, _ = tf.clip_by_global_norm(grads, 1.0)
    optimizer.apply_gradients(zip(grads, params))

    # Unscale for logging
    scale_sq = economics.REWARD_SCALE ** 2
    return loss_bellman, loss_euler, total_loss, loss_bellman / scale_sq, loss_euler / scale_sq


def get_optimal_policy_continuous(policy_net, k_batch, z_batch):
    """Inference helper."""
    inputs = tf.stack([k_batch, z_batch], axis=1)
    return tf.squeeze(policy_net(inputs))

