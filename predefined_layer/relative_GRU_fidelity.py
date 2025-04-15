from datetime import datetime

import torch

from data import data_load_and_process, new_data
from loss import get_fidelity_loss
from model import GRUPolicy, GRUValue
from plot import save_probability_animation, save_trajectory, plot_fidelity_loss
from utils import generate_layers, random_loss

if __name__ == "__main__":
    print(datetime.now())
    num_qubit = 4
    max_step = 10
    num_gate_class = 5

    num_layer = 64
    batch_size = 25
    max_episode = 8000  # 50

    lr = 0.005
    lr_val = 0.005

    temperature = 1
    discount = 0.95

    # 미리 만들 것
    layer_set = generate_layers(num_qubit, num_layer)
    X_train, X_test, Y_train, Y_test = data_load_and_process(dataset='kmnist', reduction_sz=num_qubit)
    X_train = X_train[:50]
    Y_train = Y_train[:50]

    policy_net = GRUPolicy(num_layers=num_layer)
    policy_net.train()
    value_net = GRUValue(num_layers=num_layer)
    value_net.train()

    opt = torch.optim.Adam(policy_net.parameters(), lr=lr)
    opt_val = torch.optim.Adam(value_net.parameters(), lr=lr_val)

    gate_list = None
    loss = 0
    last_fidelity_loss_list = {}
    prob_list = {}
    layer_list_list = {}
    # X1_batch, X2_batch, Y_batch = new_data(batch_size, X_train, Y_train)
    for episode in range(max_episode):
        X1_batch, X2_batch, Y_batch = new_data(batch_size, X_train, Y_train)
        random_fidelity_loss = random_loss(layer_set, max_step, X1_batch, X2_batch, Y_batch, 'fidelity')

        layer_list = []
        reward_list = []
        log_prob_list = []
        value_list = []

        for step in range(max_step):
            output = policy_net.forward(layer_list)
            prob = torch.softmax(output.squeeze() / temperature, dim=-1)

            dist = torch.distributions.Categorical(prob)
            layer_index = dist.sample()
            layer_list.append(layer_index.item())

            value = value_net(layer_list)  # shape: [1, 1]
            value_list.append(value.squeeze())

            gate_list = [item for i in layer_list for item in layer_set[int(i)]]
            fidelity_loss = get_fidelity_loss(gate_list, X1_batch, X2_batch, Y_batch)
            reward = (random_fidelity_loss - fidelity_loss) * 10

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

        print(f"Epdisode: {episode}, FidelityLoss: {fidelity_loss.item():.4f}, PolicyLoss: {policy_loss:.3f}, "
              f"ValueLoss: {value_loss:.3f}, Reward: {reward:.3f}")

        last_fidelity_loss_list[episode + 1] = {"fidelity_loss": fidelity_loss.item()}

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

    plot_fidelity_loss(last_fidelity_loss_list, '50_relative_GRU_fidelity_loss.png')
    save_probability_animation(prob_list, "50_relative_GRU_fidelity_loss_animation.mp4")
    save_trajectory(layer_list_list, filename="50_relative_GRU_fidelity_loss_trajectory.png", max_epoch_PG=max_episode,
                    num_layer=num_layer)
    print(datetime.now())
