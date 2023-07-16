import os
import random
import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image


def predict(model, test_image_name, image_transforms, idx_to_class):    # Função para prever a classe de uma só imagem
    transform = image_transforms['test']
    test_image = Image.open(test_image_name).convert('RGB')
    test_image_tensor = transform(test_image)
    if torch.cuda.is_available():
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224).cuda()
    else:
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224)

    with torch.no_grad():
        model.eval()
        out = model(test_image_tensor)
        ps = torch.exp(out)

        topk, topclass = ps.topk(3, dim=1)
        cls1 = idx_to_class[topclass.cpu().numpy()[0][0]]
        score1 = topk.cpu().numpy()[0][0]
        cls2 = idx_to_class[topclass.cpu().numpy()[0][1]]
        score2 = topk.cpu().numpy()[0][1]

        text1 = f"{cls1}: {score1 * 100:.4f}%"
        text2 = f"{cls2}: {score2 * 100:.4f}%"

        # Plot à parte para as anotações
        fig, ax = plt.subplots()
        plt.axis('off')
        # Mostrar a imagem no plot principal
        ax.imshow(test_image)

        # Adicionar os registos no plot à parte
        ax.annotate(text1, (0.5, 0.95), color='green', fontsize=12, fontweight='bold',
                    xycoords='axes fraction', ha='center', va='top',
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
        ax.annotate(text2, (0.5, 0.05), color='blue', fontsize=10,
                    xycoords='axes fraction', ha='center', va='bottom',
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

        plt.show()
        return test_image_tensor
        pass


def predict2(model, test_image_name, image_transforms, idx_to_class):   # Função para prever a classe de uma imagem enviada pela função "generate_class_images"
    transform = image_transforms['test']
    test_image = Image.open(test_image_name).convert('RGB')
    plt.imshow(test_image)
    test_image_tensor = transform(test_image)
    if torch.cuda.is_available():
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224).cuda()
    else:
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224)

    with torch.no_grad():
        model.eval()
        out = model(test_image_tensor)
        ps = torch.exp(out)

        topk, topclass = ps.topk(3, dim=1)
        cls1 = idx_to_class[topclass.cpu().numpy()[0][0]]
        score1 = topk.cpu().numpy()[0][0]
        cls2 = idx_to_class[topclass.cpu().numpy()[0][1]]
        score2 = topk.cpu().numpy()[0][1]

    return cls1, score1, cls2, score2
    pass


def generate_class_images(model, test_folder, image_transforms, idx_to_class, max_images_per_class=10):  # Função para prever a classe de várias imagens
    class_images = {}

    for root, _, files in os.walk(test_folder):
        for file in files:
            test_image_name = os.path.join(root, file)
            if os.path.isfile(test_image_name):
                cls = os.path.basename(root)  # Extração do nome da classe a partir do nome da pasta

                if cls not in class_images:
                    class_images[cls] = []

                class_images[cls].append(test_image_name)

    output_folder = "Class_Predictions"
    os.makedirs(output_folder, exist_ok=True)

    class_count = 0
    total_classes = len(class_images)

    for class_name, images in class_images.items():
        class_count += 1
        pbar = tqdm(total=len(images), desc=f"A Processar a Pasta {class_count}/{total_classes}: ")

        images_to_display = random.sample(images, min(max_images_per_class, len(images)))  # Escolher aleatoriamente imagens até prefazer 10
        num_images = len(images_to_display)
        num_rows = int((num_images + 4) / 5)  # Cálculo do número de linhas necessárias
        num_cols = min(num_images, 5)  # Limitar até ao máximo de 5 colunas

        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 8))
        fig.suptitle(class_name, fontsize=16)
        plt.subplots_adjust(wspace=0.1, hspace=0.4)  # Adjust the spacing between subplots

        for i, image_path in enumerate(images_to_display):
            row = i // num_cols
            col = i % num_cols
            img = Image.open(image_path).convert('RGB')
            img = img.resize((224, 224))

            axes[row, col].imshow(img)
            axes[row, col].axis('off')

            cls1, score1, cls2, score2 = predict2(model, image_path, image_transforms, idx_to_class)
            text1 = f"{cls1}\n{score1 * 100:.4f}%"
            text2 = f"{cls2}\n{score2 * 100:.4f}%"

            axes[row, col].text(0.5, 1.2, text1, color='green', fontsize=12, fontweight='bold',
                                transform=axes[row, col].transAxes,
                                ha='center', va='top',
                                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
            axes[row, col].text(0.5, -0.2, text2, color='blue', fontsize=10,
                                transform=axes[row, col].transAxes,
                                ha='center', va='bottom',
                                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

            pbar.update(1)  # Fazer update da progress bar

        for j in range(num_images, num_rows * num_cols):
            row = j // num_cols
            col = j % num_cols
            axes[row, col].axis('off')

        for j in range(num_images, num_rows * num_cols):
            row = j // num_cols
            col = j % num_cols
            axes[row, col].imshow(np.ones((224, 224, 3), dtype=np.uint8) * 255)
            axes[row, col].axis('off')

        plt.savefig(os.path.join(output_folder, f"{class_name}.png"))  # Guardar a imagem
        plt.close(fig)
        pbar.close()  # Fechar a progress bar

    print("Fim da Classificação das Imagens.")
    pass
