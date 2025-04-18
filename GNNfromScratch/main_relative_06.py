from datetime import datetime

import torch
from torch.functional import F
from qiskit.converters import circuit_to_dag

from data import data_load_and_process, new_data
from fidelity import check_fidelity, zz_hard_dict
from network import GNN
from plot import fidelity_plot
from representation import dict_to_qiskit_circuit, dag_to_pyg_data


def compute_gae(rewards, values, gamma=0.95, lam=0.95):
    """GAE(Generalized Advantage Estimation) 계산"""
    gae = 0
    returns = []

    # 마지막 상태의 가치 값 (에피소드 종료 시 0으로 가정)
    next_value = 0

    for i in reversed(range(len(rewards))):
        # TD 오차 계산
        delta = rewards[i] + gamma * next_value - values[i].item()

        # GAE 업데이트
        gae = delta + gamma * lam * gae

        # 다음 단계 준비
        next_value = values[i].item()

        # 리턴 값 저장 (GAE + 가치)
        returns.insert(0, gae + values[i].item())

    return torch.tensor(returns)


def calculate_entropy(distributions):
    """정책 분포들의 엔트로피 계산"""
    total_entropy = 0
    for dist in distributions:
        if dist is not None:
            entropy = -(dist * torch.log(dist + 1e-10)).sum()
            total_entropy += entropy
    return total_entropy


if __name__ == "__main__":
    start_time = datetime.now()
    num_qubit = 4
    gate_types = ["RX", "RY", "RZ", "CNOT", "H", "I"]

    batch_size = 25
    gamma = 0.95
    lam = 0.95  # GAE 람다 파라미터
    learning_rate = 0.0007
    max_episode = 20000
    max_step = 25
    entropy_coef = 0.01  # 엔트로피 계수
    hidden_dim = 64

    X_train, X_test, Y_train, Y_test = data_load_and_process(dataset='kmnist', reduction_sz=num_qubit)
    policy_net = GNN(in_dim=len(gate_types), hidden_dim=hidden_dim, num_qubit=num_qubit, num_gate_types=len(gate_types),
                     num_feature_idx=num_qubit)

    policy_net.train()
    opt = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)

    fidelity_logs = []
    try:
        for episode in range(max_episode):
            circuit_dict = [
                {'gate_type': 'H', 'depth': 0, 'qubits': (0, None), 'param': 0},
                {'gate_type': 'H', 'depth': 0, 'qubits': (1, None), 'param': 0},
                {'gate_type': 'H', 'depth': 0, 'qubits': (2, None), 'param': 0},
                {'gate_type': 'H', 'depth': 0, 'qubits': (3, None), 'param': 0},
            ]
            X1_batch, X2_batch, Y_batch = new_data(batch_size, X_train, Y_train)
            zz_fidelity_loss = check_fidelity(zz_hard_dict, X1_batch, X2_batch, Y_batch)
            print(f"zz: {zz_fidelity_loss}")

            log_probs = []
            rewards = []
            values = []

            # 정책 분포를 저장하기 위한 리스트
            entropy_dists = []

            for step in range(max_step):
                qiskit_qc = dict_to_qiskit_circuit(circuit_dict)
                dag = circuit_to_dag(qiskit_qc)
                dag_for_pyg = dag_to_pyg_data(dag, gate_types)

                q_logits, g_logits, p_logits, t_logits, value = policy_net(dag_for_pyg)

                q_dist = torch.softmax(q_logits, dim=0)
                g_dist = torch.softmax(g_logits, dim=0)
                p_dist = torch.softmax(p_logits, dim=0)
                t_dist = torch.softmax(t_logits, dim=0)

                # 엔트로피 계산을 위해 분포 저장
                step_dists = [q_dist, g_dist, None, None]

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
                    step_dists[2] = t_dist  # target qubit 분포 저장

                elif gate_name in ['RX', 'RY', 'RZ']:
                    p_sample = torch.multinomial(p_dist, num_samples=1).item()
                    qubits = (q_sample, None)
                    param = p_sample
                    log_prob = torch.log(q_dist[q_sample]) + torch.log(g_dist[g_sample]) + torch.log(p_dist[p_sample])
                    step_dists[2] = p_dist  # 파라미터 분포 저장

                else:
                    qubits = (q_sample, None)
                    param = None
                    log_prob = torch.log(q_dist[q_sample]) + torch.log(g_dist[g_sample])

                entropy_dists.append(step_dists)

                new_gate = {
                    'gate_type': gate_name,
                    'depth': step + 1,  # 현재 step을 depth로 사용
                    'qubits': qubits,
                    'param': param,
                }

                circuit_dict.append(new_gate)

                fidelity_loss = check_fidelity(circuit_dict, X1_batch, X2_batch, Y_batch)
                reward = zz_fidelity_loss.item()-fidelity_loss.item()

                log_probs.append(log_prob)
                rewards.append(reward)
                values.append(value)

            values = torch.stack(values)
            rewards = torch.tensor(rewards)

            # GAE를 사용하여 리턴 계산
            returns = compute_gae(rewards, values, gamma=gamma, lam=lam)

            # 어드밴티지 계산
            advantages = returns - values.detach()

            # 정책 손실 계산
            policy_loss = -torch.stack(log_probs) * advantages
            policy_loss = policy_loss.mean()  # sum() 대신 mean()으로 안정화

            # 가치 손실 계산
            value_loss = F.mse_loss(values, returns)

            # 모든 정책 분포에 대한 엔트로피 계산
            entropy = 0
            for step_dists in entropy_dists:
                entropy += calculate_entropy([d for d in step_dists if d is not None])
            entropy = entropy / len(entropy_dists)  # 평균 엔트로피

            # 전체 손실 계산 (정책 손실 + 가치 손실 - 엔트로피 보너스)
            loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 0.5)  # 기울기 클리핑 추가
            opt.step()

            fidelity_logs.append(fidelity_loss)
            print(f"[episode {episode}] Fidelity: {fidelity_loss:.4f}, Entropy: {entropy:.4f}")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        fidelity_plot(fidelity_logs, f"relative_step:{max_step}_lr:{learning_rate}_episode:{max_episode}.png")
        print("Time Taken:", datetime.now() - start_time)