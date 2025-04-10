import pennylane as qml
import torch

num_qubit = 4
dev = qml.device('default.qubit', wires=num_qubit)


def quantum_embedding(x, gate_list):
    for gate, qubit_idx in gate_list:
        if gate == 'R_x':
            qml.RX(x[qubit_idx], wires=qubit_idx)
        elif gate == 'R_y':
            qml.RY(x[qubit_idx], wires=qubit_idx)
        elif gate == 'R_z':
            qml.RZ(x[qubit_idx], wires=qubit_idx)
        elif gate == 'CNOT':
            qml.CNOT(wires=[qubit_idx[0], qubit_idx[1]])


@qml.qnode(dev, interface='torch')
def fidelity_circuit(x1, x2, gate_list):
    quantum_embedding(x1, gate_list)
    qml.adjoint(quantum_embedding)(x2, gate_list)
    return qml.probs(wires=range(4))


def get_fidelity_loss(gate_list, X1_batch, X2_batch, Y_batch):
    preds = []
    for x1, x2 in zip(X1_batch, X2_batch):
        probs = fidelity_circuit(x1, x2, gate_list)
        preds.append(probs[0])  # ⟨ψ₁|ψ₂⟩²

    preds = torch.stack(preds)
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(preds, Y_batch)
    return loss
