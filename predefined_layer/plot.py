import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch


def plot_policy_loss(arch_list, filename):
    x = list(arch_list.keys())
    policy_losses = [arch_list[i]['policy_loss'] for i in x]

    plt.figure(figsize=(10, 6))

    plt.plot(x, policy_losses, marker='o', linestyle='-', label='Loss')

    plt.xlabel('episode')
    plt.ylabel('Loss')
    plt.title('fidelity loss')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)  # 기준선
    plt.legend()
    plt.grid()
    plt.savefig(filename)


def save_probability_animation(prob_list, filename="animation.mp4"):
    """
    Creates an animation from a probability distribution dictionary and saves it as a video file.

    Parameters:
        prob_list (dict): A dictionary where keys represent frames and values contain 'prob' (list of probabilities).
        filename (str): Name of the output video file (default is 'animation.mp4').
    """
    # Extract frames and distributions
    frames = list(prob_list.keys())
    distributions = [prob_list[frame]['prob'] for frame in frames]

    # Initialize figure and bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_container = ax.bar(range(len(distributions[0])), distributions[0], color='skyblue', edgecolor='black')

    # Set axis properties
    ax.set_ylim(0, max(max(d) for d in distributions) * 1.2)
    ax.set_xlabel("Layer Index", fontsize=14)
    ax.set_ylabel("Probability", fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Update function for animation
    def update(frame_idx):
        for bar, height in zip(bar_container, distributions[frame_idx]):
            bar.set_height(height)
        ax.set_title(f"Probability Distribution - Epoch {frames[frame_idx]}", fontsize=16)

    # Create animation
    ani = FuncAnimation(fig, update, frames=len(frames), interval=600, repeat=True)

    # Save animation as a video file
    ani.save(filename, writer="ffmpeg")
    plt.close(fig)


def save_trajectory(li, filename="trajectory_plot.png", max_epoch_PG=200, num_layer=64):
    """
    Plots and saves a trajectory graph based on the provided data.
    Parameters:
        li (dict): Trajectory data where keys are timesteps and values contain 'layer_list'.
        filename (str): The name of the file to save the plot (default is "trajectory_plot.png").
        total_timesteps (int): Total number of timesteps to display on the x-axis.
        total_positions (int): Total number of positions (y-axis range).
    """
    colors = ['red', 'yellow', 'green', 'skyblue', 'black',
              'blue', 'orange', 'purple', 'pink', 'brown',
              'gray', 'tan', 'blue']
    # x축: timestep, y축: value
    timesteps = list(li.keys())
    trajectories = [li[t]['layer_list'] for t in timesteps]
    trajectories = preprocess_done_trajectory(trajectories)

    # 확대된 그래프 생성
    ratio = max(int(max_epoch_PG / num_layer), 1)
    plt.figure(figsize=(20 * ratio, 20))  # 커다란 그림

    for i in range(len(trajectories[0])):  # 위치 수만큼 반복
        y_values = [trajectory[i] for trajectory in trajectories]  # i번째 위치의 값 추출
        plt.plot(timesteps, y_values, marker='o', label=f'Trajectory {i + 1}', color=colors[i % len(colors)])
    # x축과 y축을 grid로 표시
    plt.xticks(range(0, max_epoch_PG + 1), fontsize=10)  # 10 간격의 xticks
    plt.yticks(range(num_layer), fontsize=10)  # 모든 y축 index 표시
    plt.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    # 그래프 속성 설정
    plt.title("Trajectory Plot", fontsize=16)
    plt.xlabel("Timestep", fontsize=14)
    plt.ylabel("Position (Layer Index)", fontsize=14)
    plt.legend(title="Trajectories", fontsize=12, title_fontsize=14)
    # 그래프 저장
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def preprocess_done_trajectory(trajectories):
    """
    길이가 다른 리스트를 가장 긴 리스트의 길이에 맞춰 마지막 값을 복사하여 확장합니다.
    """
    max_length = max(len(sublist) for sublist in trajectories)  # 가장 긴 리스트의 길이
    processed_li = []

    for sublist in trajectories:
        if len(sublist) < max_length:  # 길이가 짧으면
            last_value = torch.tensor(0)  # 마지막 값 복사 (빈 리스트인 경우 0 추가)
            sublist = sublist + [last_value] * (max_length - len(sublist))  # 길이 맞추기
        processed_li.append(sublist)

    return processed_li
