from torchvision import datasets, models
from Dataset_Split import dataSetSplit
from FeatureMaps import MakeFeatureMaps
from imageTransforms import imageTransforms
from TrainAndValidate import TrainAndValidate
from ComputeTestSetAccuracy import ComputeTestSetAccuracy
from All_Classes_Predictions import generate_class_images, predict
from ConfMatrix import generate_cm, plot_cm
from torch.utils.data import DataLoader
from torchsummary import summary
from ClearFolder import clear_folder
import torch
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from colorama import init, Fore, Style
from FineTunning import FineTunningTrainVal
init()


userInfo = -1                           # Variável para armazenar input do utilizador para saber se é para realizar treino ou testagem
testUserInfo = -1                       # Variável para armazenar input do utilizador para saber se é para testar uma imagem ou um batch de imagens
directTesting = True                    # Variável para transitar da fase de treino para a fase de testagem sem ter que carregar variáveis

while userInfo != 0 and userInfo != 1:  # While para verificar se vai ser realizado treino ou testagem
    userInfo = int(input("Pretende iniciar um novo treino (tecla 0)? Ou testar o modelo já treinado (tecla 1)?: "))
    if userInfo != 0 and userInfo != 1:
        print(Fore.RED + "### Tem que inserir o valor 0 (treinar) ou o valor 1 (testar)! ###")
        print(Style.RESET_ALL)

if userInfo == 0:                       # Caso a escolha do utilizador seja 0 vai ser realizado treino
    inputData = input("Insira o rácio (0 a 100) de treino para dividir o DataSet (Por defeito = 70): ")
    trainRatio = float(inputData)/100 if inputData != '' else 0.7
    inputData = input("Insira o rácio (0 a 100) de validação para dividir o DataSet (Por defeito = 20): ")
    validationRatio = float(inputData)/100 if inputData != '' else 0.2
    inputData = input("Insira o rácio (0 a 100) de teste para dividir o DataSet (Por defeito = 10): ")
    testRatio = float(inputData)/100 if inputData != '' else 0.1
    dataSetSplit(trainRatio, validationRatio, testRatio)    # Divisão do Dataset em sets de Treino, Validação e Testagem em função dos rácios
    inputData = input("Insira o Batch Size para o treino (Por defeito = 32): ")
    batchSize = int(inputData) if inputData != '' else 32
    inputData = input("Insira o número de épocas a realizar no treino (Por defeito = 100): ")
    numEpochs = int(inputData) if inputData != '' else 100
    image_transforms = imageTransforms()                    # As transformações que irão ser feitas aos dados

    dataset = '../TSplittedDataset_no_bg+Outras'
    train_directory = os.path.join(dataset, 'train')  # Atribuição do caminho da pasta à variável train_directory
    valid_directory = os.path.join(dataset, 'val')  # Atribuição do caminho da pasta à variável valid_directory
    test_directory = os.path.join(dataset, 'test')  # Atribuição do caminho da pasta à variável test_directory

    bs = batchSize  # Batch size

    num_classes = len(os.listdir(valid_directory))  # Obtenção do número de classes
    print(Fore.GREEN + "\t-> O DataSet tem " + str(num_classes) + " classes.")
    print(Style.RESET_ALL)

    data = {  # Carregamento dos dados a partir das pastas e aplicação das transformações
        'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
        'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid']),
        'test': datasets.ImageFolder(root=test_directory, transform=image_transforms['test'])
    }

    idx_to_class = {v: k for k, v in
                    data['train'].class_to_idx.items()}  # Dicionário com os índices e respetivas classes
    filler = "********"
    print(Fore.YELLOW + filler + "\tÍNDICE\t\t\t\t" + "CLASSE")
    for key, value in idx_to_class.items():
        print(filler + "\t  #" + str(key), "\t   -> ", "\t\t" + str(value))
    print(Style.RESET_ALL)

    train_data_size = len(
        data['train'])  # Obtenção da quantidade de dados na pasta train para poder calcular Loss e Accuracy
    valid_data_size = len(
        data['valid'])  # Obtenção da quantidade de dados na pasta valid para poder calcular Loss e Accuracy
    test_data_size = len(
        data['test'])  # Obtenção da quantidade de dados na pasta test para poder calcular Loss e Accuracy

    # Criação de iteradores para os dados através do uso do módulo DataLoader
    train_data_loader = DataLoader(data['train'], batch_size=bs, shuffle=True)
    valid_data_loader = DataLoader(data['valid'], batch_size=bs, shuffle=True)
    test_data_loader = DataLoader(data['test'], batch_size=bs, shuffle=True)

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")  # Atribuição do tipo de dispositivo em que será realizado o treino
    print("Dados de Treino = " + str(train_data_size), "\nDados de Validação = " + str(valid_data_size),
          "\nDados de Testagem = " + str(test_data_size))

    # Carregar o modelo pré-treinado ResNet50
    resnet50 = models.resnet50(weights='ResNet50_Weights.DEFAULT')

    # Congelamento dos parâmetros do modelo
    for param in resnet50.parameters():
        param.requires_grad = False

    # Obtenção dos dados originais da Fully Connected Layer da Resnet50
    fc_inputs = resnet50.fc.in_features

    # Substituição da Fully Connected Layer da ResNet-50 por uma nova
    # A nova Fully Connected Layer consiste numa linear layer, função de ativação ReLU, ...
    # ...dropout layer, outra linear layer adequada ao número de classes e uma função de ativação LogSoftmax.
    resnet50.fc = nn.Sequential(
        nn.Linear(fc_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes),
        nn.LogSoftmax(dim=1)
    )

    resnet50 = resnet50.to(device)  # Converter o modelo para poder ser treinado em CPU ou GPU
    loss_func = nn.NLLLoss()  # Definição da Loss Function
    optimizer = optim.Adam(resnet50.parameters())  # Definição do Optimizer

    # Impressão do Modelo a ser treinado
    summary(resnet50, input_size=(3, 224, 224), batch_size=bs, device='cuda' if torch.cuda.is_available() else 'cpu')

    folder_path = "Training_History"  # Pasta para guardar os ficheiros de histórico do treino
    os.makedirs(folder_path, exist_ok=True)  # Cria a pasta caso ela não exista

    # Treino do Modelo por numEpochs vezes
    trained_model, history, best_epoch = TrainAndValidate(resnet50, loss_func, optimizer, numEpochs, train_data_loader,
                                                          train_data_size, valid_data_loader, valid_data_size, device)

    ComputeTestSetAccuracy(trained_model, loss_func, test_data_loader,
                           test_data_size)  # Testar accuracy e loss para o test dataset
    MakeFeatureMaps(True, trained_model, '1.png', 0)  # Criar Feature Maps do Modelo (Igual ao original) ...
    # ...visto que nesta fase de treino não se mudam os pesos...
    # ...das camadas convolucionais (usadas para os Feature Maps).

    # Criar e salvar gráfico com os dados de treino (Perda)
    graphs_folder_path = "Graphs"  # Cria a pasta caso ela não exista
    os.makedirs(graphs_folder_path, exist_ok=True)
    history_plot = np.array(history)
    plt.plot(history_plot[:, 0:2])
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.ylim(0, 1)
    plt.savefig(graphs_folder_path + '/loss_curve.png')
    plt.show()
    plt.close()
    # Criar e salvar gráfico com os dados de treino (Acerto)
    plt.plot(history_plot[:, 2:4])
    plt.legend(['Training Accuracy', 'Validation Accuracy'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.savefig(graphs_folder_path + '/accuracy_curve.png')
    plt.show()
    plt.close()

    # Pasta para guardar os ficheiros de histórico do treino de Fine Tunning
    ft_folder_path = "Fine_Tunning_History"
    os.makedirs(ft_folder_path, exist_ok=True)  # Cria a pasta caso ela não exista
    resnet50 = torch.load("./Training_History/Grapevine Leaves Classification_model_{}.pt".format(
        best_epoch))  # Fazer load do Modelo com menores perdas
    resnet50 = resnet50.to(device)  # Atribuição do modelo ao dispositivo (CPU ou GPU)

    learning_rate = 0.0001  # Redução da taxa de aprendizagem para o Fine Tunning
    optimizer = optim.SGD(resnet50.parameters(), lr=learning_rate,
                          momentum=0.9)  # Definição do Optimizer para o Fine Tunning
    step_size = 5  # Número que define o intervalo de épocas necessárias para reduzir o Learning Rate
    gamma = 0.1  # Fator de redução do Learning Rate
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size,
                                    gamma=gamma)  # Agendador das alterações ao Learning Rate

    temp = history.tolist()  # Passa o ndarray do histórico do treino para um vetor
    ft_history = temp[:best_epoch + 1]  # Cópia do vetor acima até à melhor época do treino (inclusive)
    ft_best_epoch = 0  # Inicialização da melhor época do Fine Tunning a 0
    num_epochs_ft = 100  # Número de épocas a serem realizadas no Fine Tunning
    # Treino de Fine Tunning do Modelo por num_epochs_ft vezes
    fine_tuned_model, ft_history, ft_best_epoch = FineTunningTrainVal(resnet50, loss_func, optimizer, num_epochs_ft,
                                                                      train_data_loader, train_data_size,
                                                                      valid_data_loader,
                                                                      valid_data_size, device, ft_history, scheduler)

    ComputeTestSetAccuracy(fine_tuned_model, loss_func, test_data_loader,
                           test_data_size)  # Testar accuracy e loss para o test dataset

    # Criar e salvar gráfico com os dados de treino conjugados com o Fine Tunning (Perda)
    history_plot = np.array(history)
    ft_history = np.array(ft_history)
    plt.plot(history_plot[:, 0:2])
    plt.plot(ft_history[:, 0], color='c')
    plt.plot(ft_history[:, 1], color='r')
    plt.axvline(x=best_epoch, color='green', linestyle='--', label='Fine Tuning Start')
    plt.legend(['Training Loss', 'Validation Loss', 'FT Training Loss', 'FT Validation Loss', 'Fine Tuning Start'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.ylim(0, 1)
    plt.savefig(graphs_folder_path + '/FTunning_loss_curve.png')
    plt.show()
    plt.close()
    # Criar e salvar gráfico com os dados de treino conjugados com o Fine Tunning (Acerto)
    plt.plot(history_plot[:, 2:4])
    plt.plot(ft_history[:, 2], color='c')
    plt.plot(ft_history[:, 3], color='r')
    plt.axvline(x=best_epoch, color='green', linestyle='--', label='Fine Tuning Start')
    plt.legend(['Training Accuracy', 'Validation Accuracy', 'FT Training Accuracy', 'FT Validation Accuracy',
                'Fine Tuning Start'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.savefig(graphs_folder_path + '/FTunning_accuracy_curve.png')
    plt.show()
    plt.close()

    # Gravar os dados do ambiente de treino para um Ficheiro
    saved_data = {
        'ft_best_epoch': ft_best_epoch,
        'ft_history': ft_history,
        'best_epoch': best_epoch,
        'bs': bs,
        'data': data,
        'dataset': dataset,
        'device': device,
        'fc_inputs': fc_inputs,
        'fine_tuned_model': fine_tuned_model,
        'graphs_folder_path': graphs_folder_path,
        'history': history,
        'history_plot': history_plot,
        'idx_to_class': idx_to_class,
        'image_transforms': image_transforms,
        'learning_rate': learning_rate,
        'loss_func': loss_func,
        'num_epochs_ft': num_epochs_ft,
        'num_classes': num_classes,
        'num_epochs': num_epochs,
        'optimizer': optimizer,
        'param': param,
        'resnet50': resnet50,
        'scheduler': scheduler,
        'step_size': step_size,
        'temp': temp,
        'test_data_loader': test_data_loader,
        'test_data_size': test_data_size,
        'test_directory': test_directory,
        'ft_folder_path': ft_folder_path,
        'train_data_loader': train_data_loader,
        'train_data_size': train_data_size,
        'train_directory': train_directory,
        'trained_model': trained_model,
        'valid_data_loader': valid_data_loader,
        'valid_data_size': valid_data_size,
        'valid_directory': valid_directory
    }
    # Convert CUDA tensors to CPU tensors before saving
    for key in saved_data.keys():
        if torch.is_tensor(saved_data[key]):
            saved_data[key] = saved_data[key].to(torch.device('cpu'))
    file_path = 'saved_data.pth'
    torch.save(saved_data, file_path)
    userInfo = 1
    directTesting = False

if userInfo == 1:  # Caso a escolha do utilizador seja 1 vai ser realizada testagem
    while testUserInfo != 0 and testUserInfo != 1:
        testUserInfo = int(
            input("Pretende testar uma imagem (tecla 0), ou testar um conjunto [Test Dataset] de imagens (tecla 1)?: "))
        if testUserInfo != 0 and testUserInfo != 1:
            print(Fore.RED + "### Tem que inserir o valor 0 (uma imagem) ou o valor 1 (conjunto de imagens)! ###")
            print(Style.RESET_ALL)
    if directTesting:
        # Carregar as Variáveis
        # Verificar se o CUDA está disponivel
        if torch.cuda.is_available():
            loaded_data = torch.load('saved_data.pth', map_location=torch.device('cuda'))
        else:
            loaded_data = torch.load('saved_data.pth', map_location=torch.device('cpu'))
        ft_best_epoch = loaded_data['ft_best_epoch']
        ft_history = loaded_data['ft_history']
        best_epoch = loaded_data['best_epoch']
        bs = loaded_data['bs']
        data = loaded_data['data']
        dataset = loaded_data['dataset']
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        fc_inputs = loaded_data['fc_inputs']
        fine_tuned_model = torch.load(
            "./Fine_Tunning_History/Grapevine Leaves Classification - Fine Tunning_model_{}.pt".format(99),
            map_location=torch.device(device))  # Fine Tuned Model na última época
        graphs_folder_path = loaded_data['graphs_folder_path']
        history = loaded_data['history']
        history_plot = loaded_data['history_plot']
        idx_to_class = loaded_data['idx_to_class']
        image_transforms = loaded_data['image_transforms']
        learning_rate = loaded_data['learning_rate']
        loss_func = loaded_data['loss_func']
        num_epochs_ft = loaded_data['num_epochs_ft']
        num_classes = loaded_data['num_classes']
        num_epochs = loaded_data['num_epochs']
        optimizer = loaded_data['optimizer']
        param = loaded_data['param']
        resnet50 = loaded_data['resnet50']
        scheduler = loaded_data['scheduler']
        step_size = loaded_data['step_size']
        temp = loaded_data['temp']
        test_data_loader = loaded_data['test_data_loader']
        test_data_size = loaded_data['test_data_size']
        test_directory = loaded_data['test_directory']
        ft_folder_path = loaded_data['ft_folder_path']
        tl_model = torch.load("./Training_History/Grapevine Leaves Classification_model_{}.pt".format(best_epoch),
                              map_location=torch.device(device))
        ft_model = torch.load(
            "./Fine_Tunning_History/Grapevine Leaves Classification - Fine Tunning_model_{}.pt".format(ft_best_epoch),
            map_location=torch.device(device))
        train_data_loader = loaded_data['train_data_loader']
        train_data_size = loaded_data['train_data_size']
        train_directory = loaded_data['train_directory']
        trained_model = torch.load("./Training_History/Grapevine Leaves Classification_model_{}.pt".format(99),
                                   map_location=torch.device(device))
        valid_data_loader = loaded_data['valid_data_loader']
        valid_data_size = loaded_data['valid_data_size']
        valid_directory = loaded_data['valid_directory']

    if testUserInfo == 0:  # Caso se queira testar apenas uma imagem
        user_image = input("Insira o nome e extensão da imagem a testar (ex: \"imagem.jpg\"): ")
        test_image_tensor = predict(ft_model, user_image, image_transforms, idx_to_class)

    if testUserInfo == 1:  # Caso se queira uma pasta com várias classes de imagens
        test_folder = dataset + '/test'
        clear_folder("./Class_Predictions")
        generate_class_images(ft_model, test_folder, image_transforms, idx_to_class)

sys.exit(0)

'''######################## Secção de Código para obtenção de Dados para Executar por Linhas ########################'''
# Testar Loss e Accuracy
ComputeTestSetAccuracy(tl_model, loss_func, test_data_loader, test_data_size)
ComputeTestSetAccuracy(trained_model, loss_func, test_data_loader, test_data_size)
ComputeTestSetAccuracy(ft_model, loss_func, test_data_loader, test_data_size)
ComputeTestSetAccuracy(fine_tuned_model, loss_func, test_data_loader, test_data_size)

# Gerar a Matriz de Confusão
cm_folder_path = "Confusion Matrix"
os.makedirs(cm_folder_path, exist_ok=True)
cm1 = generate_cm(tl_model, test_data_loader, num_classes)
cm2 = generate_cm(trained_model, test_data_loader, num_classes)
cm3 = generate_cm(ft_model, test_data_loader, num_classes)
cm4 = generate_cm(fine_tuned_model, test_data_loader, num_classes)

# Fazer plot da Matriz de Confusão
class_names = np.array(list(idx_to_class.values()))
plot_cm(cm1, class_names, cm_folder_path, '#1 TL (Best Epoch)')
plot_cm(cm2, class_names, cm_folder_path, '#2 TL (Last Epoch)')
plot_cm(cm3, class_names, cm_folder_path, '#3 FT (Best Epoch)')
plot_cm(cm4, class_names, cm_folder_path, '#4 TL (Last Epoch)')
