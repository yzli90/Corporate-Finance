import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os

import economics
import model as dl_model

# =========================================================
#  Configuration
# =========================================================
NUM_EPOCHS = 30000
BATCH_SIZE = 128
PRINT_INTERVAL = 2000

K_MIN = 10.0
K_MAX = 250.0

RESULT_DIR = "results"
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)


def generate_training_data(batch_size):
    k_batch = tf.random.uniform([batch_size], minval=K_MIN, maxval=K_MAX, dtype=tf.float32)
    z_batch = tf.random.uniform([batch_size], minval=0.7, maxval=1.4, dtype=tf.float32)
    return k_batch, z_batch


def main():
    print("Initializing Maliar(21) Solver (Final Stable Version)...")
    value_net = dl_model.ValueFunctionNet()
    policy_net = dl_model.PolicyNet()

    dummy_in = tf.zeros([1, 2])
    value_net(dummy_in)
    policy_net(dummy_in)

    # --- Piecewise Constant Learning Rate Schedule ---
    # 1. [0, 12000]: 1e-3 (Fast learning)
    # 2. [12000, 24000]: 2e-4 (Refinement)
    # 3. [24000, 30000]: 1e-5 (Stability for steep Gamma)
    boundaries = [12000, 24000]
    values = [1e-3, 2e-4, 1e-5]
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries, values)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    loss_history = {'bellman_real': [], 'euler_real': []}

    print(f"Starting training for {NUM_EPOCHS} epochs...")
    start_time = time.time()

    gamma_start = 10.0
    gamma_end = 400.0

    for epoch in range(NUM_EPOCHS):
        # Gamma Schedule (Curriculum Learning)
        if epoch < 0.8 * NUM_EPOCHS:  # 0 to 24000
            progress = epoch / (0.8 * NUM_EPOCHS)
            new_gamma = gamma_start + (gamma_end - gamma_start) * progress
            economics.GAMMA_VAR.assign(new_gamma)
        else:
            economics.GAMMA_VAR.assign(gamma_end)

        k_batch, z_batch = generate_training_data(BATCH_SIZE)

        # Train Step
        l_bell, l_euler, l_total, l_bell_real, l_euler_real = dl_model.train_step_maliar_eq15(
            value_net, policy_net,
            optimizer,
            k_batch, z_batch
        )

        loss_history['bellman_real'].append(l_bell_real.numpy())
        loss_history['euler_real'].append(l_euler_real.numpy())

        if epoch % PRINT_INTERVAL == 0:
            curr_gamma = economics.GAMMA_VAR.numpy()

            curr_lr = lr_schedule(optimizer.iterations).numpy()

            print(f"Epoch {epoch} | Gamma: {curr_gamma:.0f} | LR: {curr_lr:.1e} | "
                  f"Real Bellman Err^2: {l_bell_real:.4f} | Real Euler Err^2: {l_euler_real:.4f}")

    print(f"Training done in {time.time() - start_time:.1f}s")

    # =================================================================
    #  Visualization
    # =================================================================
    print("Generating Analysis Plots...")

    plt.figure(figsize=(12, 5))

    # Plot Real Errors
    plt.subplot(1, 2, 1)

    def moving_average(a, n=100):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    bell_smooth = moving_average(np.abs(loss_history['bellman_real']))
    euler_smooth = moving_average(np.abs(loss_history['euler_real']))

    plt.plot(bell_smooth, label='Real Bellman (Smoothed)', alpha=0.8)
    plt.plot(euler_smooth, label='Real Euler (Smoothed)', alpha=0.8)
    plt.yscale('log')

    # plt.axvline(x=24000, color='r', linestyle='--', alpha=0.3, label='LR Drop (1e-5)')

    # plt.title(f"Convergence (Stable)\nFinal Bellman: {loss_history['bellman_real'][-1]:.4f}")
    plt.title("Convergence")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error (Real Units)")
    plt.legend()

    k_plot = tf.linspace(K_MIN, K_MAX, 300)
    z_fix = tf.ones_like(k_plot) * 1.0

    # Inference
    kp_pred = dl_model.get_optimal_policy_continuous(policy_net, k_plot, z_fix)

    # Compute-and-Compare Check
    flow_act = economics.shareholder_cash_flow(k_plot, z_fix, kp_pred)
    z_prime_mean = tf.math.exp(economics.RHO * tf.math.log(z_fix))
    v_next_act = value_net(tf.stack([kp_pred, z_prime_mean], axis=1))
    val_act = flow_act + economics.BETA * tf.squeeze(v_next_act)

    kp_wait = k_plot * (1.0 - economics.DELTA)
    flow_wait = economics.shareholder_cash_flow(k_plot, z_fix, kp_wait)
    v_next_wait = value_net(tf.stack([kp_wait, z_prime_mean], axis=1))
    val_wait = flow_wait + economics.BETA * tf.squeeze(v_next_wait)

    do_wait = val_wait > val_act
    final_kp = tf.where(do_wait, kp_wait, kp_pred)
    final_inv_rate = (final_kp - (1.0 - economics.DELTA) * k_plot) / k_plot

    plt.subplot(1, 2, 2)
    plt.plot(k_plot, (kp_pred - (1.0 - economics.DELTA) * k_plot) / k_plot, 'b--', alpha=0.4,
             label='Raw Network Output')
    plt.plot(k_plot, final_inv_rate, 'g-', linewidth=2, label='Corrected (S,s) Policy')
    plt.axhline(0, color='k', linestyle=':', alpha=0.5)
    plt.title("Investment Policy (z=1.0)")
    plt.xlabel("Capital k")
    plt.ylabel("Investment/k")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "maliar_final_stable.png"))
    print("Results saved.")

    # --- Additional Diagnostics (Synthetic Data) ---
    print("Generating Synthetic Data Simulation...")

    T_SIM = 100
    k_sim = [K_MIN + (K_MAX - K_MIN) / 2]
    z_sim = [1.0]
    inv_sim = []

    current_z = tf.constant([1.0], dtype=tf.float32)
    current_k = tf.constant([k_sim[0]], dtype=tf.float32)

    tf.random.set_seed(42)

    for t in range(T_SIM):
        # 1. Predict Policy k'
        kp = dl_model.get_optimal_policy_continuous(policy_net, current_k, current_z)

        # squeeze() on scalar turns (1,1) -> (). Reshape back to (1,)
        kp = tf.reshape(kp, [1])

        # 2. Apply (S, s) Check (Compute-and-Compare)
        flow_act = economics.shareholder_cash_flow(current_k, current_z, kp)
        z_prime_mean = tf.math.exp(economics.RHO * tf.math.log(current_z))

        v_next_act = value_net(tf.stack([kp, z_prime_mean], axis=1))
        val_act = flow_act + economics.BETA * tf.squeeze(v_next_act)

        # Calculate Value(Wait)
        kp_wait = current_k * (1.0 - economics.DELTA)
        flow_wait = economics.shareholder_cash_flow(current_k, current_z, kp_wait)
        v_next_wait = value_net(tf.stack([kp_wait, z_prime_mean], axis=1))
        val_wait = flow_wait + economics.BETA * tf.squeeze(v_next_wait)

        # Decision
        if val_wait > val_act:
            kp_final = kp_wait
            inv = 0.0
        else:
            kp_final = kp
            inv = (kp_final - (1.0 - economics.DELTA) * current_k).numpy()[0]

        k_val = kp_final.numpy()[0]
        k_sim.append(k_val)
        inv_sim.append(inv)

        current_k = tf.reshape(kp_final, [1])
        current_z = tf.reshape(economics.transition_shock(current_z), [1])
        z_sim.append(current_z.numpy()[0])

    # Plot Simulation
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(z_sim[:-1], 'orange')
    plt.title("Exogenous Shock (z)")
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 2)
    plt.plot(k_sim[:-1], 'b')
    plt.title("Capital Stock (k)")
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 3)
    plt.bar(range(T_SIM), inv_sim, color='green', alpha=0.6)
    plt.title("Investment (I) - Showing Lumpiness")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "synthetic_data_simulation.png"))

    # --- D. Value Function ---
    plt.figure(figsize=(6, 4))
    val_preds = value_net(tf.stack([tf.constant(k_plot, dtype=tf.float32),
                                    tf.constant(z_fix, dtype=tf.float32)], axis=1))
    plt.plot(k_plot, val_preds, 'r-', lw=2)
    plt.title("Value Function V(k, z=1.0)")
    plt.xlabel("Capital k")
    plt.ylabel("Value")
    plt.grid(True)
    plt.savefig(os.path.join(RESULT_DIR, "value_function.png"))

    # --- E. Log10 Euler Error Distribution (The "Gold Standard" for Effectiveness) ---
    print("Generating Log10 Euler Error Plot...")

    # 1. Create a dense grid of k
    k_grid = tf.linspace(K_MIN, K_MAX, 500)
    z_fix_err = tf.ones_like(k_grid) * 1.0  # Check at mean shock

    # 2. We need gradients to calculate Euler Residuals
    # Re-use the logic from train_step but without optimization
    # Note: We need to watch k_grid to compute gradients w.r.t input if needed,
    # but our Euler Residual depends on k_prime gradients.

    # To compute gradients for the plot, we can use a persistent tape
    with tf.GradientTape(persistent=True) as tape:
        # Prepare inputs
        inputs = tf.stack([k_grid, z_fix_err], axis=1)

        # Predict Policy k'
        k_prime = tf.squeeze(policy_net(inputs))
        tape.watch(k_prime)  # Watch k' to get derivatives

        # Calculate terms for Euler Equation
        # Euler: d(Reward)/dk' + beta * E[d(V')/dk'] = 0

        # 1. Marginal Reward
        reward = economics.shareholder_cash_flow(k_grid, z_fix_err, k_prime)

        # 2. Expected Marginal Continuation Value
        # Using Gaussian Quadrature or just Mean Shock for plotting approximation
        # Here we use the transition mean for simplicity in plotting
        z_prime_mean = tf.math.exp(economics.RHO * tf.math.log(z_fix_err))
        v_next = tf.squeeze(value_net(tf.stack([k_prime, z_prime_mean], axis=1)))

    # Compute Derivatives
    d_reward_dk = tape.gradient(reward, k_prime)
    d_v_next_dk = tape.gradient(v_next, k_prime)
    del tape

    # Calculate Raw Euler Residual (LHS + RHS)
    # FOC: MR + beta * MV = 0
    euler_resid = d_reward_dk + economics.BETA * d_v_next_dk

    # Normalize by Marginal Reward (to get Relative Error)
    # Avoid division by zero
    denom = tf.abs(d_reward_dk) + 1e-6
    rel_error = tf.abs(euler_resid) / denom

    # Log10 Error
    log10_err = tf.math.log(rel_error + 1e-10) / tf.math.log(10.0)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(k_grid, log10_err, 'purple', alpha=0.7, linewidth=1)
    plt.title("Log10 Euler Equation Errors (Effectiveness Metric)")
    plt.xlabel("Capital Stock k")
    plt.ylabel("Log10 Relative Error")
    plt.axhline(-2, color='r', linestyle='--', alpha=0.3, label='1% Error')
    plt.axhline(-3, color='g', linestyle='--', alpha=0.3, label='0.1% Error')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Highlight the Inaction Region (approx)
    # In Inaction Region, Euler Eq doesn't hold (Inequality), so errors might be high.
    plt.text(20, -1.5, "Action Region", fontsize=10)
    plt.text(150, -1.5, "Inaction Region\n(Euler Eq not binding)", fontsize=10)

    plt.savefig(os.path.join(RESULT_DIR, "euler_error_distribution.png"))
    print("All diagnostics generated.")


if __name__ == "__main__":
    main()

