import pennylane as qml
import torch

dev = qml.device('default.qubit', wires=4)


def apply_gate(gate, x):
    gate_type = gate['gate_type']
    param_idx = gate['param']
    param_value = x[param_idx]

    ###########수정된 부분##########
    # qubit index 정의
    qubits = gate['qubits']
    if gate_type == 'CNOT':
        control, target = qubits
    else:
        target = qubits[0]
    ###########수정된 부분##########

    # gate 적용
    if gate_type == 'RX':
        qml.RX(param_value, wires=target)
    elif gate_type == 'RY':
        qml.RY(param_value, wires=target)
    elif gate_type == 'RZ':
        qml.RZ(param_value, wires=target)
    elif gate_type == 'H':
        qml.Hadamard(wires=target)
    elif gate_type == 'CNOT':
        qml.CNOT(wires=[control, target])
    elif gate_type == 'I':
        pass


def apply_circuit(x, circuit):
    for gate in sorted(circuit, key=lambda g: g['depth']):
        apply_gate(gate, x)


@qml.qnode(dev, interface='torch')
def fidelity_circuit(x1, x2, circuit):
    apply_circuit(x1, circuit)
    qml.adjoint(apply_circuit)(x2, circuit)
    return qml.probs(wires=range(4))


def check_fidelity(circuit, X1_batch, X2_batch, Y_batch):
    preds = []
    for x1, x2 in zip(X1_batch, X2_batch):
        probs = fidelity_circuit(x1, x2, circuit)
        preds.append(probs[0])  # ⟨ψ₁|ψ₂⟩²

    preds = torch.stack(preds)
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(preds, Y_batch)
    return loss


def check_triplet_fidelity_loss(circuit, Xa, Xp, Xn, margin=0.1):
    losses = []
    for xa, xp, xn in zip(Xa, Xp, Xn):
        # 모든 fidelity는 [0,1] 사이, fidelity_circuit은 pure state overlap
        f_pos = fidelity_circuit(xa, xp, circuit)[0]
        f_neg = fidelity_circuit(xa, xn, circuit)[0]
        loss = torch.clamp(f_neg - f_pos + margin, min=0.0)
        losses.append(loss)
    return torch.stack(losses).mean()


@qml.qnode(dev, interface='torch')
def output_probs(x, circuit):
    apply_circuit(x, circuit)
    return qml.probs(wires=range(4))


def similarity_score(x1, x2, circuit):
    p1 = output_probs(x1, circuit)
    p2 = output_probs(x2, circuit)
    tvd = 0.5 * torch.sum(torch.abs(p1 - p2))  # Total Variation Distance
    return 1.0 - tvd  # similarity


def check_rep_cap_loss(circuit, X_batch, Y_batch):
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
