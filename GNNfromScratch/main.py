from datetime import datetime

import torch
from qiskit.converters import circuit_to_dag

from data import data_load_and_process, new_data
from fidelity import check_fidelity
from network import GNN
from plot import fidelity_plot
from representation import dict_to_qiskit_circuit, dag_to_pyg_data


if __name__ == "__main__":
    print(datetime.now())
    num_qubit = 4
    gate_types = ["RX", "RY", "RZ", "CNOT", "H", "I"]

    batch_size = 25
    gamma = 0.95
    learning_rate = 0.0003
    max_episode = 300
    max_step = 10

    hidden_dim = 64

    X_train, X_test, Y_train, Y_test = data_load_and_process(dataset='kmnist', reduction_sz=num_qubit)
    policy_net = GNN(in_dim=len(gate_types), hidden_dim=hidden_dim, num_qubit=num_qubit, num_gate_types=len(gate_types),
                     num_feature_idx=num_qubit)

    policy_net.train()
    opt = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)

    fidelity_logs = []

    for episode in range(max_episode):
        # done = False
        circuit_dict = [
            {'gate_type': 'H', 'depth': 0, 'qubits': (0, None), 'param': 0},
            {'gate_type': 'H', 'depth': 0, 'qubits': (1, None), 'param': 0},
            {'gate_type': 'H', 'depth': 0, 'qubits': (2, None), 'param': 0},
            {'gate_type': 'H', 'depth': 0, 'qubits': (3, None), 'param': 0},
        ]
        X1_batch, X2_batch, Y_batch = new_data(batch_size, X_train, Y_train)

        log_probs = []
        rewards = []

        for step in range(max_step):
            print(f"#######################step:{step}#######################")
            print(circuit_dict)

            qiskit_qc = dict_to_qiskit_circuit(circuit_dict)
            dag = circuit_to_dag(qiskit_qc)
            dag_for_pyg = dag_to_pyg_data(dag, gate_types)

            # try:
            #     qiskit_qc = dict_to_qiskit_circuit(circuit_dict)
            #     dag = circuit_to_dag(qiskit_qc)
            #     dag_for_pyg = dag_to_pyg_data(dag, gate_types)
            # except Exception as e:
            #     print(f"[Ep {episode}, Step {step}] Transpile or DAG error: {e}")
            #     fidelity_loss = torch.tensor(1.0)
            #     fidelity_logs.append(fidelity_loss)
            #     rewards.append(-1.0)
            #     log_probs.append(torch.tensor(0.0))  # dummy log_prob
            #     done = True
            #     break

            q_logits, g_logits, p_logits, t_logits = policy_net(dag_for_pyg)

            q_dist = torch.softmax(q_logits, dim=0)
            g_dist = torch.softmax(g_logits, dim=0)
            p_dist = torch.softmax(p_logits, dim=0)
            t_dist = torch.softmax(t_logits, dim=0)

            q_sample = torch.multinomial(q_dist, num_samples=1).item()
            g_sample = torch.multinomial(g_dist, num_samples=1).item()

            gate_name = gate_types[g_sample]

            if gate_name == 'CNOT':
                t_sample_offset = torch.multinomial(t_dist, num_samples=1).item()
                t_sample = (q_sample + t_sample_offset + 1) % num_qubit
                qubits = (q_sample, t_sample)
                param = None
                log_prob = torch.log(q_dist[q_sample]) + torch.log(g_dist[g_sample]) + torch.log(
                    t_dist[t_sample_offset])

            elif gate_name in ['RX', 'RY', 'RZ']:
                p_sample = torch.multinomial(p_dist, num_samples=1).item()
                qubits = (q_sample, None)
                param = p_sample
                log_prob = torch.log(q_dist[q_sample]) + torch.log(g_dist[g_sample]) + torch.log(p_dist[p_sample])

            else:
                qubits = (q_sample, None)
                param = None
                log_prob = torch.log(q_dist[q_sample]) + torch.log(g_dist[g_sample])

            new_gate = {
                'gate_type': gate_name,
                'depth': step + 1,  # 현재 step을 depth로 사용
                'qubits': qubits,
                'param': param,
            }
            # if circuit_dict and new_gate == circuit_dict[-1]:
            #     print(f"[warn] duplicated gate at step {step}: {new_gate}")

            circuit_dict.append(new_gate)

            fidelity_loss = check_fidelity(circuit_dict, X1_batch, X2_batch, Y_batch)
            reward = -fidelity_loss.item()
            # try:
            #     fidelity_loss = check_fidelity(circuit_dict, X1_batch, X2_batch, Y_batch)
            #     reward = -fidelity_loss.item()
            # except Exception as e:
            #     print(f"[Ep {episode}, Step {step}] Fidelity eval error: {e}")
            #     fidelity_loss = torch.tensor(1.0)
            #     reward = -1.0
            #     done = True

            log_probs.append(log_prob)
            rewards.append(reward)

            # if done:
            #     break

        returns = torch.tensor(rewards)
        baseline = returns.mean()
        advantages = returns - baseline
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        loss = 0
        for log_prob, A in zip(log_probs, advantages):
            loss -= log_prob * A

        opt.zero_grad()
        loss.backward()
        opt.step()

        fidelity_logs.append(fidelity_loss)
        print(f"[episode {episode}] Fidelity: {fidelity_loss:.4f}")

    fidelity_plot(fidelity_logs, "fidelity.png")
