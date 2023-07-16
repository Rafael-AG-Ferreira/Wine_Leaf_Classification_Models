import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.metrics import confusion_matrix


# Função para gerar a Matriz de Confusão
def generate_cm(model, dataloader, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_predictions, labels=range(num_classes))

    return cm
    pass


# Função para fazer Plot da Matriz de Confusão
def plot_cm(cm, class_names, folder_path, filename):
    fig, ax = plt.subplots()
    # Ajustar o brilho das cores através da normalização
    cmap = plt.colormaps['coolwarm']
    norm = Normalize(vmin=-0.1, vmax=0.2 * np.max(cm))

    im = ax.imshow(cm, cmap=cmap, norm=norm)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=6)
    ax.set_yticklabels(class_names, fontsize=6)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(class_names)):
        row_sum = np.sum(cm[i])
        true_positives = cm[i, i]
        false_negatives = row_sum - true_positives
        percentage = true_positives / row_sum * 100 if row_sum != 0 else 0
        row_text = f"{row_sum:.0f} = {true_positives:.0f} + {false_negatives:.0f} ({percentage:.1f}%)"
        ax.text(len(class_names), i, row_text, ha="left", va="center", fontsize=8, color="black", fontweight="bold")
        for j in range(len(class_names)):
            text = ax.text(j, i, format(cm[i, j]),
                           ha="center", va="center",
                           color="white" if cm[i, j] > cm.max() / 2 else "black",
                           fontsize=8)

    ax.set_title("Matriz de Confusão")
    ax.set_xlabel("Classes Previstas")
    ax.set_ylabel("Classes Reais")

    # Colocar os nomes das classes nas porções externas
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    fig.tight_layout()
    plt.savefig(folder_path + '/' + filename, dpi=600)
    plt.show()
    pass
