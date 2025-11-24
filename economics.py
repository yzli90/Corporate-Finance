import tensorflow as tf

"""
Economic primitives for Strebulaev (2012).
Defines production, costs, and shock processes.
"""

# --- Model Parameters (Section 3.1) ---
THETA = 0.7  # Returns to scale
DELTA = 0.1  # Depreciation
R = 0.04  # Risk-free rate
BETA = 1.0 / (1.0 + R)

# Shock process (AR1)
RHO = 0.7
SIGMA_E = 0.15

# Adjustment costs
PSI_0 = 0.05  # Convex cost
PSI_1 = 0.01  # Fixed cost

# Scaling for numerical stability
REWARD_SCALE = 0.01
# Gamma for smoothing the fixed cost indicator (increases during training)
GAMMA_VAR = tf.Variable(50.0, dtype=tf.float32, trainable=False)


def production_function(k, z):
    """Profit = z * k^theta"""
    k = tf.cast(k, dtype=tf.float32)
    z = tf.cast(z, dtype=tf.float32)
    return z * tf.pow(k, THETA)


def get_investment(k, k_prime):
    """Implied investment: I = k' - (1 - delta) * k"""
    return k_prime - (1.0 - DELTA) * k


def adjustment_cost(i, k):
    """
    Cost = Convex + Fixed.
    Uses a smooth approximation for the fixed cost indicator (I != 0).
    """
    # Convex part
    convex = (PSI_0 / 2.0) * tf.square(i) / (k + 1e-8)

    # Fixed part (approximated)
    fixed = 0.0
    if PSI_1 > 0.0:
        # 1 - exp(-gamma * I^2) approximates the indicator function
        smooth_indicator = 1.0 - tf.exp(-GAMMA_VAR * tf.square(i))
        fixed = PSI_1 * k * smooth_indicator

    return convex + fixed


def shareholder_cash_flow(k, z, k_prime):
    """Net payout = Profit - Cost - Investment"""
    i_val = get_investment(k, k_prime)
    profit = production_function(k, z)
    adj_cost = adjustment_cost(i_val, k)

    return (profit - adj_cost - i_val) * REWARD_SCALE


def transition_shock(z, seed=None):
    """Next period shock z' (Log-normal AR1)"""
    ln_z = tf.math.log(z + 1e-8)
    noise = tf.random.normal(shape=tf.shape(z), mean=0.0, stddev=SIGMA_E, seed=seed)
    ln_z_next = RHO * ln_z + noise
    return tf.math.exp(ln_z_next)

