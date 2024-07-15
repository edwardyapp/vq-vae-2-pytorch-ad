import json
import matplotlib.pyplot as plt

# Load loss values from JSON files
with open('train_losses.json', 'r') as f:
    train_losses = json.load(f)
with open('val_losses.json', 'r') as f:
    val_losses = json.load(f)

# Plotting the loss curves
epochs = range(1,  + 1)
plt.plot(epochs, train_losses['recon_loss'], label='Training Recon Loss')
plt.plot(epochs, val_losses['recon_loss'], label='Validation Recon Loss')
# Plot other loss curves similarly
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Reconstruction Loss')
plt.legend()
plt.show()
