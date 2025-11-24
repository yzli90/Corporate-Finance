import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import economics
import model as dl_model

# Config
NUM_EPOCHS = 30000
BATCH_SIZE = 128
RESULT_DIR = "results_maliar"
if not os.path.exists(RESULT_DIR): os.makedirs(RESULT_DIR)


def generate_data(bs):
    k = tf.random.uniform([bs], 10.0, 250.0)
    z = tf.random.uniform([bs], 0.7, 1.4)
    return k, z


def main():
    print("Setting up model...")
    v_net = dl_model.ValueFunctionNet()
    p_net = dl_model.PolicyNet()
    # Build shapes
    v_net(tf.zeros([1, 2]));
    p_net(tf.zeros([1, 2]))

    # LR Schedule: High -> Low to settle convergence
    lr_sch = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        [12000, 24000], [1e-3, 2e-4, 1e-5])
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_sch)

    history = {'bellman': [], 'euler': []}
    start = time.time()

    # Curriculum for fixed cost approximation (Gamma)
    g_start, g_end = 10.0, 400.0

    print("Training...")
    for epoch in range(NUM_EPOCHS):
        # Update Gamma
        prog = min(epoch / (0.8 * NUM_EPOCHS), 1.0)
        economics.GAMMA_VAR.assign(g_start + (g_end - g_start) * prog)

        k_b, z_b = generate_data(BATCH_SIZE)

        # Train step
        _, _, _, l_b_real, l_e_real = dl_model.train_step_maliar_eq15(
            v_net, p_net, optimizer, k_b, z_b)

        history['bellman'].append(l_b_real.numpy())
        history['euler'].append(l_e_real.numpy())

        if epoch % 2000 == 0:
            lr = lr_sch(optimizer.iterations).numpy()
            print(f"Ep {epoch} | G: {economics.GAMMA_VAR.numpy():.0f} | "
                  f"LR: {lr:.1e} | Bellman: {l_b_real:.4f} | Euler: {l_e_real:.4f}")

    print(f"Done. Time: {time.time() - start:.1f}s")

    # --- Analysis & Plotting ---

    # 1. Convergence & Policy
    plt.figure(figsize=(12, 5))

    # Loss Plot
    plt.subplot(1, 2, 1)

    def smooth(x):
        return np.convolve(x, np.ones(100) / 100, mode='valid')

    plt.plot(smooth(np.abs(history['bellman'])), label='Bellman')
    plt.plot(smooth(np.abs(history['euler'])), label='Euler')
    plt.yscale('log')
    plt.legend();
    plt.title("Convergence")

    # Policy Plot with (S,s) correction
    k_plot = tf.linspace(10.0, 250.0, 300)
    z_fix = tf.ones_like(k_plot)

    # NN raw output
    kp_nn = dl_model.get_optimal_policy_continuous(p_net, k_plot, z_fix)

    # Compute "Wait" option value to enforce (S,s) region
    # Value of acting
    flow_act = economics.shareholder_cash_flow(k_plot, z_fix, kp_nn)
    z_next = tf.math.exp(economics.RHO * tf.math.log(z_fix))
    v_next_act = v_net(tf.stack([kp_nn, z_next], axis=1))
    val_act = flow_act + economics.BETA * tf.squeeze(v_next_act)

    # Value of waiting (invest = 0)
    kp_wait = k_plot * (1.0 - economics.DELTA)
    flow_wait = economics.shareholder_cash_flow(k_plot, z_fix, kp_wait)
    v_next_wait = v_net(tf.stack([kp_wait, z_next], axis=1))
    val_wait = flow_wait + economics.BETA * tf.squeeze(v_next_wait)

    # Choose max
    final_kp = tf.where(val_wait > val_act, kp_wait, kp_nn)
    inv_rate = (final_kp - (1.0 - economics.DELTA) * k_plot) / k_plot

    plt.subplot(1, 2, 2)
    plt.plot(k_plot, (kp_nn - (1.0 - economics.DELTA) * k_plot) / k_plot, 'b--', alpha=0.4, label='Raw NN')
    plt.plot(k_plot, inv_rate, 'g-', lw=2, label='Corrected (S,s)')
    plt.axhline(0, color='k', ls=':')
    plt.legend();
    plt.title("Investment Policy")
    plt.savefig(os.path.join(RESULT_DIR, "results.png"))

    # 2. Synthetic Simulation
    T_SIM = 100
    curr_k, curr_z = tf.constant([120.0]), tf.constant([1.0])
    hist_k, hist_z, hist_i = [], [], []

    for _ in range(T_SIM):
        # Prediction
        kp = dl_model.get_optimal_policy_continuous(p_net, curr_k, curr_z)
        kp = tf.reshape(kp, [1])

        # Check Wait vs Act (Simplified logic for simulation loop)
        z_n = tf.math.exp(economics.RHO * tf.math.log(curr_z))
        v_act = economics.shareholder_cash_flow(curr_k, curr_z, kp) + \
                economics.BETA * tf.squeeze(v_net(tf.stack([kp, z_n], axis=1)))

        kp_w = curr_k * (1.0 - economics.DELTA)
        v_wait = economics.shareholder_cash_flow(curr_k, curr_z, kp_w) + \
                 economics.BETA * tf.squeeze(v_net(tf.stack([kp_w, z_n], axis=1)))

        if v_wait > v_act:
            kp_final = kp_w
            inv = 0.0
        else:
            kp_final = kp
            inv = (kp_final - (1.0 - economics.DELTA) * curr_k).numpy()[0]

        hist_k.append(kp_final.numpy()[0])
        hist_z.append(curr_z.numpy()[0])
        hist_i.append(inv)

        curr_k = tf.reshape(kp_final, [1])
        curr_z = tf.reshape(economics.transition_shock(curr_z), [1])

    plt.figure()
    plt.subplot(311);
    plt.plot(hist_z);
    plt.title("Shock z")
    plt.subplot(312);
    plt.plot(hist_k);
    plt.title("Capital k")
    plt.subplot(313);
    plt.bar(range(T_SIM), hist_i);
    plt.title("Lumpy Investment")
    plt.tight_layout();
    plt.savefig(os.path.join(RESULT_DIR, "sim.png"))

    # 3. Log10 Euler Errors
    k_g = tf.linspace(10.0, 250.0, 500)
    z_g = tf.ones_like(k_g)

    with tf.GradientTape() as t:
        kp = tf.squeeze(p_net(tf.stack([k_g, z_g], axis=1)))
        t.watch(kp)
        rew = economics.shareholder_cash_flow(k_g, z_g, kp)
        zn = tf.math.exp(economics.RHO * tf.math.log(z_g))
        vn = tf.squeeze(v_net(tf.stack([kp, zn], axis=1)))

    dr_dk = t.gradient(rew, kp)
    dv_dk = t.gradient(vn, kp)
    err = dr_dk + economics.BETA * dv_dk

    rel_err = tf.abs(err) / (tf.abs(dr_dk) + 1e-6)
    log_err = tf.math.log(rel_err + 1e-10) / tf.math.log(10.0)

    plt.figure()
    plt.plot(k_g, log_err)
    plt.axhline(-2, color='r', ls='--', label='1%')
    plt.title("Log10 Euler Errors")
    plt.legend();
    plt.savefig(os.path.join(RESULT_DIR, "errors.png"))
    print("Diagnostics done.")


if __name__ == "__main__":
    main()