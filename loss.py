import matplotlib.pyplot as plt

epoch_losses = [6.12, 4.89, 3.91, 3.21, 2.74, 2.58, 2.44, 2.35, 2.28, 2.22,
                2.18, 2.15, 2.13, 2.11, 2.10, 2.09, 2.08, 2.08, 2.17, 2.17]

plt.figure(figsize=(10, 5))
plt.plot(range(1, 21), epoch_losses, marker='o', color='steelblue', linewidth=2)
plt.title("Training Loss over 20 Epochs", fontsize=14)
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=300)
plt.show()