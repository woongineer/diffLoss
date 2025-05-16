import torch
from qiskit import QuantumCircuit, transpile
from torch_geometric.data import Data


def dict_to_qiskit_circuit(circuit_dict):
    max_qubit = max(q for g in circuit_dict for q in g['qubits'] if q is not None)
    qc = QuantumCircuit(max_qubit + 1)
    for gate in sorted(circuit_dict, key=lambda g: g['depth']):
        name = gate['gate_type']
        qubits = [q for q in gate['qubits'] if q is not None]
        if name == 'CNOT':
            qc.cx(*qubits)
        elif name == 'RX':
            qc.rx(0.5, qubits[0])
        elif name == 'RY':
            qc.ry(0.5, qubits[0])
        elif name == 'RZ':
            qc.rz(0.5, qubits[0])
        elif name == 'H':
            qc.h(qubits[0])
    return transpile(qc, basis_gates=["rx", "ry", "rz", "cx", "h"], optimization_level=1)

def dag_to_pyg_data(dag, gate_types):
    def encode_gate_type(name):
        mapping = {
            "rx": "RX",
            "ry": "RY",
            "rz": "RZ",
            "cx": "CNOT",
            "h": "H",
        }
        name = mapping.get(name, name.upper())
        vec = torch.zeros(len(gate_types))
        if name in gate_types:
            vec[gate_types.index(name)] = 1
        return vec

    node_feats = []
    node_id_map = {}
    edge_index = []

    op_nodes = list(dag.topological_op_nodes())
    if not op_nodes:
        one_hot = torch.zeros(len(gate_types))
        one_hot[gate_types.index('I')] = 1
        depth_scalar = torch.tensor([0.0])  # dummy라도 7차원 맞춰야 함
        dummy_x = torch.cat([one_hot, depth_scalar]).unsqueeze(0)
        return Data(x=dummy_x, edge_index=torch.empty((2, 0), dtype=torch.long))
        # raise ValueError("dag_to_pyg_data: DAG has no operation nodes. Check if circuit was optimized to empty.")

    for i, node in enumerate(op_nodes):
        node_id_map[node] = i
        one_hot = encode_gate_type(node.name)

        ###########수정된 부분 START##########
        # depth / max_depth  (0‒1 정규화)  추가
        depth_scalar = torch.tensor([node._node_id / dag.depth()], dtype=torch.float32)
        node_feats.append(torch.cat([one_hot, depth_scalar]))

    for i, node in enumerate(op_nodes):
        for succ in dag.successors(node):
            if succ in node_id_map:
                edge_index.append([node_id_map[node], node_id_map[succ]])

    x = torch.stack(node_feats)
    if edge_index:
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    else:
        edge_index_tensor = torch.empty((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index_tensor)
