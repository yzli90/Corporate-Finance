import tensorflow as tf
import numpy as np
import pytest
import economics
import model as dl_model


#  Configuration for Tests
@pytest.fixture(scope="module")
def setup_models():
    """
    Fixture to initialize models once for use in multiple tests.
    Returns value_net, policy_net
    """
    value_net = dl_model.ValueFunctionNet()
    policy_net = dl_model.PolicyNet()

    # Build models with dummy input
    dummy_in = tf.zeros([1, 2])
    value_net(dummy_in)
    policy_net(dummy_in)

    return value_net, policy_net


# =========================================================
#  Part 1: Unit Tests
#  Ensure basic economic functions return correct values
# =========================================================

def test_production_function():
    """Test if production function z * k^theta calculates correctly."""
    k = tf.constant([100.0])
    z = tf.constant([1.0])

    # Expected: 1.0 * 100^0.7 = 25.1188
    expected = 1.0 * (100.0 ** 0.7)
    result = economics.production_function(k, z).numpy()[0]

    assert np.isclose(result, expected, atol=1e-4), "Production function calculation mismatch"


def test_adjustment_cost_zero_investment():
    """
    Test that adjustment cost is zero (or very close to it) when investment is zero.
    Critical for checking the smooth approximation logic.
    """
    k = tf.constant([100.0])
    i = tf.constant([0.0])  # Zero investment

    # When I=0, convex cost is 0. 
    # Fixed cost uses smooth indicator: 1 - exp(-gamma * 0) = 1 - 1 = 0.
    cost = economics.adjustment_cost(i, k).numpy()[0]

    assert np.isclose(cost, 0.0, atol=1e-4), "Adjustment cost should be 0 when Investment is 0"


def test_adjustment_cost_high_investment():
    """Test that fixed cost triggers when investment is non-zero."""
    k = tf.constant([100.0])
    i = tf.constant([5.0])  # Significant investment

    cost = economics.adjustment_cost(i, k).numpy()[0]

    # Convex part: (0.05/2) * 25 / 100 = 0.00625
    # Fixed part: 0.01 * 100 * (1 - exp(-large)) ≈ 1.0
    # Total should be > 1.0
    assert cost > 1.0, "Fixed adjustment cost failed to trigger on significant investment"


def test_transition_shock_shape():
    """Test that AR(1) shock generator returns correct shape."""
    batch_size = 10
    z = tf.ones([batch_size], dtype=tf.float32)
    z_next = economics.transition_shock(z)

    assert z_next.shape == (batch_size,), "Transition shock output shape is incorrect"
    assert tf.reduce_all(z_next > 0), "Shocks must be positive (log-normal)"


# =========================================================
#  Part 2: Integration Tests
#  Ensure neural networks connect and gradients flow
# =========================================================

def test_model_output_shapes(setup_models):
    """Test that Value and Policy networks return correct output shapes."""
    value_net, policy_net = setup_models
    batch_size = 32

    k_batch = tf.random.uniform([batch_size], 10, 100)
    z_batch = tf.random.uniform([batch_size], 0.7, 1.3)
    inputs = tf.stack([k_batch, z_batch], axis=1)

    v_out = value_net(inputs)
    k_prime_out = policy_net(inputs)

    assert v_out.shape == (batch_size, 1), "Value Net output shape mismatch"
    assert k_prime_out.shape == (batch_size, 1), "Policy Net output shape mismatch"


def test_train_step_gradient_update(setup_models):
    """
    CRITICAL TEST: Run one training step and verify that weights actually change.
    This proves the gradient tape is working and the pipeline is connected.
    """
    value_net, policy_net = setup_models
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # 1. Get weights before training
    initial_weights_v = [w.numpy() for w in value_net.trainable_variables]

    # 2. Run one training step
    k_batch = tf.random.uniform([32], 10, 100)
    z_batch = tf.random.uniform([32], 0.7, 1.3)

    loss_bellman, loss_euler, _, _, _ = dl_model.train_step_maliar_eq15(
        value_net, policy_net, optimizer, k_batch, z_batch
    )

    # 3. Check for NaN
    assert not np.isnan(loss_bellman), "Bellman loss returned NaN"
    assert not np.isnan(loss_euler), "Euler loss returned NaN"

    # 4. Verify weights changed
    final_weights_v = [w.numpy() for w in value_net.trainable_variables]

    # Check at least one weight layer has changed
    weights_changed = False
    for w_init, w_final in zip(initial_weights_v, final_weights_v):
        if not np.allclose(w_init, w_final):
            weights_changed = True
            break

    assert weights_changed, "Gradients failed to update model weights!"


# =========================================================
#  Part 3: Effectiveness/Validation Tests (Logic Check)
#  Verify the Euler Equation Error calculation pipeline
#  Test the calculation logic
# =========================================================

def test_euler_error_calculation_logic(setup_models):
    """
    Test the logic for computing Log10 Euler Errors.
    This corresponds to the 'Effectiveness' requirement in the interview.
    """
    value_net, policy_net = setup_models

    # Create a small test grid
    k_grid = tf.linspace(10.0, 100.0, 50)
    z_fix = tf.ones_like(k_grid) * 1.0

    # Replicate the logic from main2.py to calculate Euler Residuals
    with tf.GradientTape(persistent=True) as tape:
        inputs = tf.stack([k_grid, z_fix], axis=1)
        k_prime = tf.squeeze(policy_net(inputs))
        tape.watch(k_prime)

        reward = economics.shareholder_cash_flow(k_grid, z_fix, k_prime)
        z_prime_mean = tf.math.exp(economics.RHO * tf.math.log(z_fix))
        v_next = tf.squeeze(value_net(tf.stack([k_prime, z_prime_mean], axis=1)))

    d_reward_dk = tape.gradient(reward, k_prime)
    d_v_next_dk = tape.gradient(v_next, k_prime)
    del tape

    # Calculate Euler Residual
    euler_resid = d_reward_dk + economics.BETA * d_v_next_dk

    # Assertions
    assert d_reward_dk is not None, "Gradient of Reward w.r.t k' is None (Graph disconnected)"
    assert d_v_next_dk is not None, "Gradient of V_next w.r.t k' is None (Graph disconnected)"
    assert euler_resid.shape == (50,), "Euler residual shape mismatch"

    # Check that relative error calculation doesn't crash
    denom = tf.abs(d_reward_dk) + 1e-6
    rel_error = tf.abs(euler_resid) / denom
    log10_err = tf.math.log(rel_error + 1e-10) / tf.math.log(10.0)

    assert not np.any(np.isnan(log10_err)), "Log10 Euler Errors contain NaNs"


if __name__ == "__main__":
    # Allow running directly via python test_project.py
    pytest.main([__file__])