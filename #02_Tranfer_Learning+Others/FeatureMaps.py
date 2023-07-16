import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from torchvision import models, transforms
from tqdm import tqdm
from PIL import Image


def MakeFeatureMaps(flag_first, model, f_m_image_path, model_epoch):    # Função para criar os Feature Maps
    # Normalizar o Dataset
    transform = transforms.Compose([
        transforms.Resize(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Fazer load de uma imagem
    image = Image.open(str(f_m_image_path)).convert('RGB')

    # Declaração do Modelo Pré Treinado
    if flag_first:
        initialModel = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        model_epoch = 'ORIGINAL'
    else:
        initialModel = model

    # A abordagem padrão é ver os feature depois das camadas Conv2d porque é nesta camada que os filtros são aplicados
    # Gravar os pesos da camada conv na lista
    model_weights = []
    conv_layers = []
    model_children = list(initialModel.children())
    counter = 0
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter += 1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter += 1
                        model_weights.append(child.weight)
                        conv_layers.append(child)

    # Verificar a possibilidade de utilizar GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    initialModel = initialModel.to(device)

    # Aplicar as tranformações na Imagem e carregar a mesma no Device
    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(device)

    # Processar a imagem para cada camada e juntar o output e nome da camada ao outputs[] e names[]
    outputs = []
    names = []
    for layer in conv_layers[0:]:
        image = layer(image)
        outputs.append(image)
        names.append(str(layer))

    # Converter o tensor 3D em 2D
    processed = []
    for feature_map in outputs:
        feature_map = feature_map.squeeze(0)
        gray_scale = torch.sum(feature_map,0)
        gray_scale = gray_scale / feature_map.shape[0]
        processed.append(gray_scale.data.cpu().numpy())


    # Criar a pasta "... + Nome Recebido + ..." caso esta não exista
    folder_name = "Trained Model(" + str(model_epoch) + ") Feature Maps"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Gravar a imagem com todos os Feature Maps
    fig = plt.figure(figsize=(30, 50))
    for i in tqdm(range(len(processed)), desc='A criar a Imagem com Todas as Camadas'):
        a = fig.add_subplot(10, 10, i + 1)
        imgplot = plt.imshow(processed[i])
        a.axis("off")
        a.set_title(names[i].split('(')[0] + " #" + str(i), fontsize=20)

    plt.savefig(os.path.join(folder_name, "all_conv_layers(" + str(model_epoch) + ")_feature_maps.jpg"),
                bbox_inches='tight', dpi=100)

    # Gravar cada um dos FEature Maps Individualmente
    for i in tqdm(range(len(processed)), desc='A criar as Imagens das Camadas Convolucionais'):
        subfig = plt.figure(figsize=(5, 5))
        subax = subfig.add_subplot(1, 1, 1)
        imgplot = subax.imshow(processed[i])
        subax.axis("off")
        subax.set_title(names[i].split('(')[0] + " #" + str(i), fontsize=10)

        subimage_path = os.path.join(folder_name, names[i].split('(')[0] + " (" + str(model_epoch) + ")" + " #" + str(i) + ".jpg")
        plt.savefig(subimage_path, bbox_inches='tight', dpi=100)
        plt.close(subfig)  # Close the figure after saving the subimage
    plt.close()
    pass
