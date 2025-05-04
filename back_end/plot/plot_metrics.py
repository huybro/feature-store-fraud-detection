import matplotlib.pyplot as plt
import numpy as np

# Define metrics and corresponding scores
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
baseline_scores = [0.9740, 0.9600, 0.8822, 0.9194] #collected results
featurestore_scores = [0.9714, 0.9666, 0.8564, 0.9080] #collected results

# Set bar positions
x = np.arange(len(metrics))  # Label locations
width = 0.35  # Width of the bars

# Create plot
fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width/2, baseline_scores, width, label='Baseline', color='cornflowerblue')
bars2 = ax.bar(x + width/2, featurestore_scores, width, label='Feature Store', color='seagreen')

# Set labels and title
ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0.8, 1.0)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.6)

# Add value annotations on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

plt.tight_layout()
plt.show()
