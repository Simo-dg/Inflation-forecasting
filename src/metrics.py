import matplotlib.pyplot as plt

def plot_model_comparison(rmse_dict, title="Model RMSE Comparison", filename="model_comparison.png"):
    # order models by RMSE
    sorted_items = sorted(rmse_dict.items(), key=lambda x: x[1])
    models = [k for k, _ in sorted_items]
    errors = [v for _, v in sorted_items]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(models, errors, color='steelblue')

    # Add labels
    for bar, err in zip(bars, errors):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{err:.2f}",
                 ha='center', va='bottom', fontsize=9)

    plt.ylabel("RMSE")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"results/{filename}")
    plt.close()
