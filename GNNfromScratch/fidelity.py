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


zz_hard_dict = [
    # ===== L=1 (첫 번째 레이어) =====
    # 1. Feature Encoding (각 큐빗에 H + RZ)
    {'gate_type': 'H', 'depth': 0, 'qubits': (0, None), 'param': 0},
    {'gate_type': 'RZ', 'depth': 1, 'qubits': (0, None), 'param': 0},

    {'gate_type': 'H', 'depth': 2, 'qubits': (1, None), 'param': 1},
    {'gate_type': 'RZ', 'depth': 3, 'qubits': (1, None), 'param': 1},

    {'gate_type': 'H', 'depth': 4, 'qubits': (2, None), 'param': 2},
    {'gate_type': 'RZ', 'depth': 5, 'qubits': (2, None), 'param': 2},

    {'gate_type': 'H', 'depth': 6, 'qubits': (3, None), 'param': 3},
    {'gate_type': 'RZ', 'depth': 7, 'qubits': (3, None), 'param': 3},

    # 2. Entanglement 블록: for k in [0,1,2]
    # Pair (0,1)
    {'gate_type': 'CNOT', 'depth': 8, 'qubits': (0, 1), 'param': None},
    {'gate_type': 'RZ', 'depth': 9, 'qubits': (1, None), 'param': 1},
    {'gate_type': 'CNOT', 'depth': 10, 'qubits': (0, 1), 'param': None},

    # Pair (1,2)
    {'gate_type': 'CNOT', 'depth': 11, 'qubits': (1, 2), 'param': None},
    {'gate_type': 'RZ', 'depth': 12, 'qubits': (2, None), 'param': 2},
    {'gate_type': 'CNOT', 'depth': 13, 'qubits': (1, 2), 'param': None},

    # Pair (2,3)
    {'gate_type': 'CNOT', 'depth': 14, 'qubits': (2, 3), 'param': None},
    {'gate_type': 'RZ', 'depth': 15, 'qubits': (3, None), 'param': 3},
    {'gate_type': 'CNOT', 'depth': 16, 'qubits': (2, 3), 'param': None},

    # Pair (3,0) (wrap-around)
    {'gate_type': 'CNOT', 'depth': 17, 'qubits': (3, 0), 'param': None},
    {'gate_type': 'RZ', 'depth': 18, 'qubits': (0, None), 'param': 0},
    {'gate_type': 'CNOT', 'depth': 19, 'qubits': (3, 0), 'param': None},

    # ===== L=2 (두 번째 레이어) =====
    # 1. Feature Encoding (각 큐빗에 H + RZ)
    {'gate_type': 'H', 'depth': 20, 'qubits': (0, None), 'param': 0},
    {'gate_type': 'RZ', 'depth': 21, 'qubits': (0, None), 'param': 0},

    {'gate_type': 'H', 'depth': 22, 'qubits': (1, None), 'param': 1},
    {'gate_type': 'RZ', 'depth': 23, 'qubits': (1, None), 'param': 1},

    {'gate_type': 'H', 'depth': 24, 'qubits': (2, None), 'param': 2},
    {'gate_type': 'RZ', 'depth': 25, 'qubits': (2, None), 'param': 2},

    {'gate_type': 'H', 'depth': 26, 'qubits': (3, None), 'param': 3},
    {'gate_type': 'RZ', 'depth': 27, 'qubits': (3, None), 'param': 3},

    # 2. Entanglement 블록: for k in [0,1,2]
    # Pair (0,1)
    {'gate_type': 'CNOT', 'depth': 28, 'qubits': (0, 1), 'param': None},
    {'gate_type': 'RZ', 'depth': 29, 'qubits': (1, None), 'param': 1},
    {'gate_type': 'CNOT', 'depth': 30, 'qubits': (0, 1), 'param': None},

    # Pair (1,2)
    {'gate_type': 'CNOT', 'depth': 31, 'qubits': (1, 2), 'param': None},
    {'gate_type': 'RZ', 'depth': 32, 'qubits': (2, None), 'param': 2},
    {'gate_type': 'CNOT', 'depth': 33, 'qubits': (1, 2), 'param': None},

    # Pair (2,3)
    {'gate_type': 'CNOT', 'depth': 34, 'qubits': (2, 3), 'param': None},
    {'gate_type': 'RZ', 'depth': 35, 'qubits': (3, None), 'param': 3},
    {'gate_type': 'CNOT', 'depth': 36, 'qubits': (2, 3), 'param': None},

    # Pair (3,0) (wrap-around)
    {'gate_type': 'CNOT', 'depth': 37, 'qubits': (3, 0), 'param': None},
    {'gate_type': 'RZ', 'depth': 38, 'qubits': (0, None), 'param': 0},
    {'gate_type': 'CNOT', 'depth': 39, 'qubits': (3, 0), 'param': None},
]
