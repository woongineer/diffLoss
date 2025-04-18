import matplotlib.pyplot as plt


def fidelity_plot(fidelity_logs, filename):
    plt.figure(figsize=(10, 4))
    plt.plot(fidelity_logs, marker='o')
    plt.title("Fidelity Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Fidelity Loss (1 - Fidelity)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
