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



def get_QPMeL_loss(gate_list, Xa_batch, Xp_batch, Xn_batch, margin=0.1):
    losses = []
    for xa, xp, xn in zip(Xa_batch, Xp_batch, Xn_batch):
        f_pos = fidelity_circuit(xa, xp, gate_list)[0]
        f_neg = fidelity_circuit(xa, xn, gate_list)[0]
        loss = torch.clamp(f_neg - f_pos + margin, min=0.0)
        losses.append(loss)
    return torch.stack(losses).mean()


@qml.qnode(dev, interface='torch')
def output_probs(x, circuit):
    quantum_embedding(x, circuit)
    return qml.probs(wires=range(4))


def similarity_score(x1, x2, circuit):
    p1 = output_probs(x1, circuit)
    p2 = output_probs(x2, circuit)
    tvd = 0.5 * torch.sum(torch.abs(p1 - p2))  # Total Variation Distance
    return 1.0 - tvd  # similarity


def get_RepCap_loss(circuit, X_batch, Y_batch):
    N = len(X_batch)
    R_C = torch.zeros((N, N))
    R_ref = torch.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i <= j:  # symmetric, 계산 절반만
                sim = similarity_score(X_batch[i], X_batch[j], circuit)
                R_C[i, j] = sim
                R_C[j, i] = sim

            # label 비교는 regardless
            R_ref[i, j] = 1.0 if Y_batch[i] == Y_batch[j] else 0.0

    # Frobenius norm-based RepCap loss (작을수록 좋음)
    norm_diff = torch.norm(R_C - R_ref, p='fro') ** 2
    nc = 2
    dc = N // nc
    norm_factor = 2 * nc * (dc ** 2)
    loss = norm_diff / norm_factor
    return loss