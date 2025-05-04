import matplotlib.pyplot as plt

# Data for training time
training_times = [527.63, 118.38] #collected results
training_labels = ['Baseline', 'Feature Store']

# Data for inference time
inference_times = [25.61, 0.99] #collected results
inference_labels = ['Baseline', 'Feature Store']

# Plot training time comparison
plt.figure(figsize=(7, 5))
bars = plt.bar(training_labels, training_times, color=['steelblue', 'mediumseagreen'])
plt.title('Average Training Time Comparison')
plt.ylabel('Time (seconds)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 10, f'{yval:.2f}', ha='center', va='bottom')
plt.tight_layout()
plt.show()

# Plot inference time comparison
plt.figure(figsize=(7, 5))
bars = plt.bar(inference_labels, inference_times, color=['salmon', 'mediumorchid'])
plt.title('Average Inference Time Comparison')
plt.ylabel('Time (seconds)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 1, f'{yval:.2f}', ha='center', va='bottom')
plt.tight_layout()
plt.show()
