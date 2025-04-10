import torch
from datetime import datetime

from data import data_load_and_process, new_data
from model import CNNLSTMPolicy, CNNLSTMValue
from utils import generate_layers, make_arch
from loss import get_fidelity_loss
from plot import save_probability_animation, save_trajectory, plot_policy_loss

if __name__ == "__main__":
    print(datetime.now())
    num_qubit = 4
    max_step = 3
    num_gate_class = 5

    num_layer = 64
    batch_size = 25
    max_episode = 4  # 50

    lr = 0.002
    lr_val = 0.002

    temperature = 0.5
    discount = 0.9

    # 미리 만들 것
    layer_set = generate_layers(num_qubit, num_layer)
    X_train, X_test, Y_train, Y_test = data_load_and_process(dataset='kmnist', reduction_sz=num_qubit)

    policy_net = CNNLSTMPolicy(feature_dim=16, hidden_dim=32, output_dim=num_layer, num_layers=1)
    policy_net.train()
    value_net = CNNLSTMValue(feature_dim=16, hidden_dim=32)
    value_net.train()

    opt = torch.optim.Adam(policy_net.parameters(), lr=lr)
    opt_val = torch.optim.Adam(value_net.parameters(), lr=lr_val)

    gate_list = None
    loss = 0
    arch_list = {}
    prob_list = {}
    layer_list_list = {}
    for episode in range(max_episode):
        X1_batch, X2_batch, Y_batch = new_data(batch_size, X_train, Y_train)

        layer_list = []
        reward_list = []
        log_prob_list = []
        value_list = []

        current_arch = torch.randint(0, 1, (1, 1, num_qubit, num_gate_class)).float()

        for step in range(max_step):
            output = policy_net.forward(current_arch)
            prob = torch.softmax(output.squeeze() / temperature, dim=-1)

            dist = torch.distributions.Categorical(prob)
            layer_index = dist.sample()
            layer_list.append(layer_index)

            value = value_net(current_arch)  # shape: [1, 1]
            value_list.append(value.squeeze())

            gate_list = [item for i in layer_list for item in layer_set[int(i)]]
            current_arch = make_arch(gate_list, num_qubit)

            fidelity_loss = get_fidelity_loss(gate_list, X1_batch, X2_batch, Y_batch)
            reward = 1 - fidelity_loss

            log_prob = dist.log_prob(layer_index.clone().detach())
            log_prob_list.append(log_prob)
            reward_list.append(reward)

        layer_list_list[episode + 1] = {'layer_list': layer_list}
        prob_list[episode + 1] = {'prob': prob.detach().tolist()}

        returns = []
        G = 0
        for r in reversed(reward_list):
            G = r + discount * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)

        values = torch.stack(value_list)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        log_prob_tensor = torch.stack(log_prob_list)
        policy_loss = -(log_prob_tensor * advantages.detach()).mean()
        value_loss = advantages.pow(2).mean()
        total_loss = policy_loss + value_loss

        print(f"Epdisode: {episode}, LastStepFidelityLoss: {fidelity_loss.item():.4f}, "
              f"PolicyLoss: {policy_loss:.3f}, ValueLoss: {value_loss:.3f}")

        arch_list[episode + 1] = {"policy_loss": policy_loss.item(), "gate_list": gate_list}

        opt.zero_grad()
        opt_val.zero_grad()

        total_loss.backward()

        total_norm = 0.0
        for p in policy_net.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=1.0)

        opt.step()
        opt_val.step()

    plot_policy_loss(arch_list, 'CNNLSTM_fidelity_loss.png')
    save_probability_animation(prob_list, "CNNLSTM_fidelity_loss_animation.mp4")
    save_trajectory(layer_list_list, filename="CNNLSTM_fidelity_loss_trajectory.png", max_epoch_PG=max_episode, num_layer=num_layer)
    print(datetime.now())

