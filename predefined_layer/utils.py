import random

import numpy as np
import torch
from loss import get_fidelity_loss, get_QPMeL_loss


def generate_layers(num_qubit, num_layers):
    single_qubit_gates = ["R_x", "R_y", "R_z"]
    layer_dict = {}
    generated_layers = set()  # To track unique layers

    for layer_idx in range(num_layers):
        while True:
            # Step 1: Randomly assign single-qubit gates
            single_gates = []
            qubits_for_single = random.sample(range(num_qubit), num_qubit // 2)
            for qubit in qubits_for_single:
                gate = random.choice(single_qubit_gates)
                single_gates.append((gate, qubit))

            # Step 2: Randomly assign CNOT gates
            cnot_gates = []
            qubits_for_cnot = random.sample(range(num_qubit), num_qubit)
            for i in range(0, len(qubits_for_cnot), 2):  # Pair qubits for CNOT
                control, target = qubits_for_cnot[i], qubits_for_cnot[i + 1]
                cnot_gates.append(("CNOT", (control, target)))

            # Step 3: Combine single and CNOT gates
            layer = single_gates + cnot_gates

            # Step 4: Ensure layer is functionally unique
            layer_tuple = tuple(sorted(layer))  # Sort for uniqueness
            if layer_tuple not in generated_layers:
                generated_layers.add(layer_tuple)
                layer_dict[layer_idx] = layer
                break  # Exit the loop once a unique layer is found

    return layer_dict


def make_arch(layer_list_flat, num_qubit):
    arch = np.zeros((1, len(layer_list_flat), num_qubit, 5))
    for time, (gate, qubit_idx) in enumerate(layer_list_flat):
        if gate == 'R_x':
            arch[0, time, qubit_idx, 0] = 1
        elif gate == 'R_y':
            arch[0, time, qubit_idx, 1] = 1
        elif gate == 'R_z':
            arch[0, time, qubit_idx, 2] = 1
        elif gate == 'CNOT':
            arch[0, time, qubit_idx[0], 3] = 1
            arch[0, time, qubit_idx[1], 4] = 1

    return torch.from_numpy(arch).float()


def make_arch_sb3(layer_list_flat, num_qubit, max_layer_step, num_gate_class):
    arch = torch.zeros((len(layer_list_flat), num_qubit, num_gate_class))
    for time, (gate, qubit_idx) in enumerate(layer_list_flat):
        if gate == 'R_x':
            arch[time, qubit_idx, 0] = 1
        elif gate == 'R_y':
            arch[time, qubit_idx, 1] = 1
        elif gate == 'R_z':
            arch[time, qubit_idx, 2] = 1
        elif gate == 'CNOT':
            arch[time, qubit_idx[0], 3] = 1
            arch[time, qubit_idx[1], 4] = 1

    padded_arch = torch.zeros(max_layer_step * 4, num_qubit, num_gate_class)
    padded_arch[:arch.shape[0], :arch.shape[1], :arch.shape[2]] = arch

    return padded_arch


def random_loss(layer_set, max_step, X1_or_Xa, X2_or_Xp, Y_or_Xn, loss_function_name):
    num_layer = len(layer_set)
    random_layer_list = random.choices(range(num_layer), k=max_step)
    random_gate_list = [item for i in random_layer_list for item in layer_set[int(i)]]
    if loss_function_name == 'fidelity':
        loss = get_fidelity_loss(random_gate_list, X1_or_Xa, X2_or_Xp, Y_or_Xn)
    elif loss_function_name == 'QPMeL':
        loss = get_QPMeL_loss(random_gate_list, X1_or_Xa, X2_or_Xp, Y_or_Xn)
    else:
        raise ValueError("Invalid loss function name. Choose either 'fidelity' or 'QPMeL'.")
    return loss