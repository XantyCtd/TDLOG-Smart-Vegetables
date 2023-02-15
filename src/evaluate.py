import matplotlib.pyplot as plt
import json

# Visualize and training results
history = json.load(open("data/2_model/history.json", "r"))

acc = history["accuracy"]
val_acc = history["val_accuracy"]

epochs = len(acc)

loss = history["loss"]
val_loss = history["val_loss"]

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")


plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.title("Training and Validation Loss")
plt.savefig("data/3_reporting/Training and Validation.png")
