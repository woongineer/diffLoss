from datetime import datetime
from collections import deque
import argparse

import torch
from torch.functional import F
from qiskit.converters import circuit_to_dag

from fidelity import zz_hard_dict, check_fidelity
from data import data_load_and_process, new_data
from network import GNN
from plot import fidelity_plot
from representation import dict_to_qiskit_circuit, dag_to_pyg_data


# Vectorized GAE
def compute_gae(rewards, values, gamma=0.95, lam=0.95):
    values = values.squeeze()
    deltas = rewards + gamma * torch.cat([values[1:], values[-1:].zero_()]) - values
    gae, returns = 0.0, torch.zeros_like(rewards)
    for t in reversed(range(len(rewards))):
        gae = deltas[t] + gamma * lam * gae
        returns[t] = gae + values[t]
    return returns.detach()


# Policy‑distribution entropy
def calculate_entropy(dists):
    return -(dists * torch.log(dists + 1e-10)).sum()


# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    start = datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--lam", type=float, default=0.95, help="GAE lambda")
    args = parser.parse_args()

    num_qubit = 4
    gate_types = ["RX", "RY", "RZ", "CNOT", "H", "I"]
    in_dim = len(gate_types) + 1            # depth 스칼라 포함

    ###########수정된 부분 START##########
    # Hyper‑params
    batch_size = 128                         # ↑ 배치로 variance 저감
    max_episode, max_step = 20000, 15
    gamma, lam = args.gamma, args.lam

    # Reward 스케일 / baseline
    reward_scale = 10.0
    baseline_buffer = deque(maxlen=500)     # running baseline

    # LR & entropy warm‑up
    lr_init = 5e-4
    entropy_coef_init, entropy_coef_min = 0.05, 0.005
    warmup_ep = 5000
    epsilon_greedy = 0.02                   # 2 % explorer
    ###########수정된 부분 END##########

    hidden_dim = 64

    X_train, X_test, Y_train, Y_test = data_load_and_process("kmnist", reduction_sz=num_qubit,
                                                             train_len=800, test_len=200)

    policy_net = GNN(in_dim, hidden_dim,
                     num_qubit=num_qubit,
                     num_gate_types=len(gate_types),
                     num_feature_idx=num_qubit).train()

    opt = torch.optim.AdamW(policy_net.parameters(), lr=lr_init)

    fid_logs, zz_logs = [], []
    eps_const = 1e-6

    try:
        for ep in range(max_episode):
            # -------- entropy coefficient warm‑up --------
            entropy_coef = max(
                entropy_coef_init * (1 - ep / warmup_ep),
                entropy_coef_min
            )

            # -------- 초기 회로 & 데이터 샘플 --------
            circuit = [{'gate_type': 'H', 'depth': 0, 'qubits': (q, None), 'param': 0}
                       for q in range(num_qubit)]
            X1, X2, Y = new_data(batch_size, X_train, Y_train)

            zz_loss = check_fidelity(zz_hard_dict, X1, X2, Y)

            log_probs, rewards, values, entropies = [], [], [], []

            for step in range(max_step):
                dag = circuit_to_dag(dict_to_qiskit_circuit(circuit))
                pyg_data = dag_to_pyg_data(dag, gate_types)

                q_log, g_log, p_log, t_log, val = policy_net(pyg_data)
                d_q = torch.softmax(q_log, 0)
                d_g = torch.softmax(g_log, 0)
                d_p = torch.softmax(p_log, 0)
                d_t = torch.softmax(t_log, 0)

                # ---------- ε‑greedy exploration ----------
                def mix(dist, n):
                    return (1 - epsilon_greedy) * dist + epsilon_greedy / n

                d_q, d_g = mix(d_q, num_qubit), mix(d_g, len(gate_types))
                d_p, d_t = mix(d_p, num_qubit), mix(d_t, num_qubit - 1)

                q = torch.multinomial(d_q, 1).item()
                g = torch.multinomial(d_g, 1).item()
                gate = gate_types[g]

                if gate == "CNOT":
                    t_off = torch.multinomial(d_t, 1).item()
                    tgt = (q + t_off + 1) % num_qubit
                    qubits, param = (q, tgt), None
                    log_p = torch.log(d_q[q]) + torch.log(d_g[g]) + torch.log(d_t[t_off])
                    entropies.append(d_t)
                elif gate in {"RX", "RY", "RZ"}:
                    p = torch.multinomial(d_p, 1).item()
                    qubits, param = (q, None), p
                    log_p = torch.log(d_q[q]) + torch.log(d_g[g]) + torch.log(d_p[p])
                    entropies.append(d_p)
                else:
                    qubits, param = (q, None), None
                    log_p = torch.log(d_q[q]) + torch.log(d_g[g])

                circuit.append({'gate_type': gate,
                                'depth': step + 1,
                                'qubits': qubits,
                                'param': param})

                fid_loss = check_fidelity(circuit, X1, X2, Y)

                ###########수정된 부분 START##########
                raw_r = (zz_loss - fid_loss) / (abs(zz_loss) + eps_const)
                baseline_buffer.append(raw_r.item())
                baseline = sum(baseline_buffer) / len(baseline_buffer)
                reward = reward_scale * (raw_r - baseline)
                ###########수정된 부분 END##########

                log_probs.append(log_p)
                rewards.append(reward)
                values.append(val)

            # -------- A2C update --------
            values = torch.stack(values)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            returns = compute_gae(rewards, values, gamma, lam)
            adv = returns - values.detach()

            ###########수정된 부분 START##########
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)   # 표준화
            ###########수정된 부분 END##########

            policy_loss = -(torch.stack(log_probs) * adv).mean()
            value_loss = F.mse_loss(values.squeeze(), returns)

            entropy_term = torch.stack([calculate_entropy(e) for e in entropies]).mean()
            loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy_term

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 0.5)
            opt.step()

            fid_logs.append(fid_loss)
            zz_logs.append(zz_loss)

            if ep % 5 == 0:
                print(f"[Ep {ep:5d}] Fid {fid_loss:.4f} | ZZ {zz_loss:.4f} | R {reward:.4f} | Ent {entropy_term:.3f}")

    except Exception as err:
        print("Error:", err)

    finally:
        fidelity_plot(fid_logs, f"fid_loss_bs{batch_size}_gamma{gamma}_lam{lam}.png")
        fidelity_plot(zz_logs,  f"zz_bs{batch_size}_gamma{gamma}_lam{lam}.png")
        print("Total time:", datetime.now() - start)
