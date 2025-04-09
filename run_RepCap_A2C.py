from copy import deepcopy
from datetime import datetime

import torch

from analysis import fidelity_plot
from data import data_load_and_process, new_data, get_class_balanced_batch
from fidelity import check_rep_cap_loss, check_fidelity
from initializer import initialize_circuit
from modifier import remover, inserter
from policy import PolicyInsertWithMask, PolicyRemove, insert_gate_map, ValueNetwork
from utils import fill_identity_gates, representer, FixedLinearProjection, sample_remove_position, \
    sample_insert_gate_param, ordering

if __name__ == "__main__":
    print(datetime.now())

    num_of_qubit = 4
    gate_types = ["RX", "RY", "RZ", "CNOT", "H"]
    depth = 10

    representation_dim = 64
    policy_dim = 128
    batch_size = 64
    gamma = 0.95
    learning_rate = 0.0003
    max_episode = 800
    max_step = 25
    fidelity_drop_threshold = 0.5

    X_train, X_test, Y_train, Y_test = data_load_and_process(dataset='kmnist', reduction_sz=num_of_qubit)
    circuit_original = initialize_circuit("random", depth, num_of_qubit, gate_types)  # either 'zz' or 'random'
    circuit_original = fill_identity_gates(circuit_original, num_of_qubit, depth)

    tensor = representer(circuit_info=circuit_original, num_qubits=num_of_qubit, depth=depth, gate_types=gate_types)
    projection = FixedLinearProjection(in_dim=tensor.shape[1], out_dim=representation_dim)

    policy_remove = PolicyRemove(input_dim=representation_dim, hidden_dim=policy_dim)
    policy_insert = PolicyInsertWithMask(input_dim=representation_dim, hidden_dim=policy_dim)
    value_net = ValueNetwork(input_dim=representation_dim, hidden_dim=policy_dim)

    policy_remove.train()
    policy_insert.train()
    value_net.train()

    opt_remove = torch.optim.Adam(policy_remove.parameters(), lr=learning_rate)
    opt_insert = torch.optim.Adam(policy_insert.parameters(), lr=learning_rate)
    opt_value = torch.optim.Adam(value_net.parameters(), lr=learning_rate)

    fidelity_logs = []

    for episode in range(max_episode):
        done = False
        circuit = deepcopy(circuit_original)
        X1_batch, X2_batch, Y_batch = new_data(batch_size, X_train, Y_train)
        X_repcap, Y_repcap = get_class_balanced_batch(X_train, Y_train, dc=16)

        low_fidelity_steps = 0
        initial_fidelity = check_fidelity(circuit, X1_batch, X2_batch, Y_batch)
        fidelity_threshold = initial_fidelity * fidelity_drop_threshold

        log_probs_list = []
        advantages_list = []
        value_loss_list = []

        for step in range(max_step):
            # print(f"step {step}")
            # plot_circuit(circuit, num_of_qubit)
            # print("#########")
            circuit_representation = representer(circuit, num_of_qubit, depth, gate_types)
            dense_representation = projection(circuit_representation)

            # 현재 상태의 가치
            value_s = value_net(dense_representation)
            # (B=1, 1) 형태라면 스칼라로 만들어주자
            value_s = value_s.squeeze(0).squeeze(-1)

            # 1) REMOVE
            remove_prob_tensor = policy_remove(dense_representation)
            qubit_index, depth_index, remove_log_prob = sample_remove_position(remove_prob_tensor)

            circuit_removed = remover(circuit, qubit_index, depth_index)
            circuit_removed_representation = representer(circuit_removed, num_of_qubit, depth, gate_types)
            dense_removed_representation = projection(circuit_removed_representation)

            # 2) INSERT
            insert_prob_tensor = policy_insert(dense_removed_representation, qubit_index, depth_index)
            insert_decision, insert_log_prob = sample_insert_gate_param(insert_prob_tensor, insert_gate_map,
                                                                        qubit_index, num_of_qubit, circuit_removed,
                                                                        depth_index)

            circuit_inserted = inserter(circuit_removed, depth_index, insert_decision)
            circuit_inserted = ordering(circuit_inserted)
            circuit_inserted = fill_identity_gates(circuit_inserted, num_of_qubit, depth)
            inserted_fidelity = check_fidelity(circuit_inserted, X1_batch, X2_batch, Y_batch)
            inserted_loss = check_rep_cap_loss(circuit_inserted, X_repcap, Y_repcap)

            reward = -inserted_loss.item()
            circuit = circuit_inserted

            # 3) next state 가치
            next_representation = representer(circuit, num_of_qubit, depth, gate_types)
            dense_next_representation = projection(next_representation)
            value_s_next = value_net(dense_next_representation)
            value_s_next = value_s_next.squeeze(0).squeeze(-1)

            # done 체크
            if inserted_fidelity <= fidelity_threshold:
                low_fidelity_steps += 1
            else:
                low_fidelity_steps = 0

            if low_fidelity_steps >= 10:
                done = True
                print("done activated")

            # 4) Advantage 계산
            if done or step == max_step - 1:
                advantage = reward - value_s  # 마지막 or done이면 V(s')=0 가정
            else:
                advantage = reward + gamma * value_s_next - value_s

            log_prob_sum = remove_log_prob + insert_log_prob
            log_probs_list.append(log_prob_sum)
            advantages_list.append(advantage)

            # critic의 loss(= (advantage)^2)를 저장
            value_loss_list.append(advantage.pow(2))

            if done:
                break

        ###########수정된 부분##########
        # A2C: policy_loss, value_loss 구하기
        log_probs_tensor = torch.stack(log_probs_list)
        advantages_tensor = torch.stack(advantages_list).detach()  # advantage는 critic 학습에 의해 업데이트되므로 detach 안 해도 가능
        value_loss_tensor = torch.stack(value_loss_list)

        policy_loss = - (log_probs_tensor * advantages_tensor).mean()
        value_loss = value_loss_tensor.mean()

        # 총 loss = policy_loss + critic_loss
        loss = policy_loss + value_loss

        # 옵티마이저 업데이트
        opt_remove.zero_grad()
        opt_insert.zero_grad()
        opt_value.zero_grad()

        loss.backward()

        opt_remove.step()
        opt_insert.step()
        opt_value.step()

        fidelity_logs.append(inserted_fidelity)
        print(f"[episode {episode}] Fidelity: {inserted_fidelity:.4f}, Reward: {reward:.4f}")

    fidelity_plot(fidelity_logs, "fidelity_RepCap_A2C.png")
