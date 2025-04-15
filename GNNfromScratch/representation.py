import torch
from qiskit import QuantumCircuit, transpile
from torch_geometric.data import Data


def dict_to_qiskit_circuit(circuit_dict):
    max_qubit = max(max(q for q in g['qubits'] if q is not None) for g in circuit_dict)
    qc = QuantumCircuit(max_qubit + 1)
    for gate in circuit_dict:
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
    return transpile(qc, optimization_level=1)

def dag_to_pyg_data(dag, gate_types):
    def encode_gate_type(name):
        vec = torch.zeros(len(gate_types))
        if name in gate_types:
            vec[gate_types.index(name)] = 1
        return vec

    node_feats = []
    node_id_map = {}
    edge_index = []

    op_nodes = list(dag.topological_op_nodes())
    if not op_nodes:
        raise ValueError("dag_to_pyg_data: DAG has no operation nodes. Check if circuit was optimized to empty.")

    for i, node in enumerate(op_nodes):
        node_id_map[node] = i
        node_feats.append(encode_gate_type(node.name))

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
