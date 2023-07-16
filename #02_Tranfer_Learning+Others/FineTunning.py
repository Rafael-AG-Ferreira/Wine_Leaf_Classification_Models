import torch
import time
from tqdm import tqdm
from colorama import init, Fore, Style
from FeatureMaps import MakeFeatureMaps
init()


# Função para Fine Tunning
def FineTunningTrainVal(model, loss_criterion, optimizer, epochs, train_data_loader, train_data_size, valid_data_loader,
                     valid_data_size, device, history, scheduler):
    best_loss = 100000.0            # Inicialização do best_loss
    best_epoch = 100000000000       # Inicialização do best_epoch

    for epoch in range(epochs):
        epoch_start = time.time()
        print(Fore.CYAN + "Fine Tunning - Época: {}/{}".format(epoch + 1, epochs))

        train_loss = 0.0            # Inicialização do training loss
        train_acc = 0.0             # Inicialização do training accuracy
        valid_loss = 0.0            # Inicialização do validation loss
        valid_acc = 0.0             # Inicialização do validation accuracy

        # Descongelamento Gradual
        if epoch == 5:
            for param in model.layer4.parameters():
                param.requires_grad = True
                model.layer4.training = True
        if epoch == 8:
            for param in model.layer3.parameters():
                param.requires_grad = True
                model.layer3.training = True
        if epoch == 10:
            for param in model.layer2.parameters():
                param.requires_grad = True
                model.layer2.training = True
        if epoch == 12:
            for param in model.layer1.parameters():
                param.requires_grad = True
                model.layer1.training = True
        # Descongelamento da Global Average Pooling Layer (Penúltima Camada)
        model.avgpool.requires_grad_(True)
        model.avgpool.training = True


        model.train()
        # Criação da Barra de Progresso do Treino
        progress_bar_training = tqdm(enumerate(train_data_loader), total=len(train_data_loader),
                                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}] \033[92m')

        for i, (inputs, labels) in progress_bar_training:
            inputs = inputs.to(device)                  # Move os dados do tensor para o device (cpu neste caso) para serem manipulados como inputs
            labels = labels.to(device)                  # Move os dados do tensor para o device (cpu neste caso) para serem manipulados como labels
            optimizer.zero_grad()                       # Zerar os gradientes antes do próximo batch
            outputs = model(inputs)                     # Forward pass -> computar os outputs nos dados de input
            loss = loss_criterion(outputs, labels)      # Computar as perdas
            loss.backward()                             # Retropropagação dos gradientes
            optimizer.step()                            # Atualizar os parâmetros
            train_loss += loss.item() * inputs.size(0)  # Computar as perdas totais para o batch e adicioná-las ao train_loss
            ret, predictions = torch.max(outputs.data, 1)                       # Encontra o valor máximo e o seu índice na dimensão 1 do tensor outputs.data
            correct_counts = predictions.eq(labels.data.view_as(predictions))   # Compara as labels previstas com as reais e determina as predições corretas
            acc = torch.mean(correct_counts.type(torch.FloatTensor))            # Converte o correct_counts para float e computa a média para obter o acerto
            train_acc += acc.item() * inputs.size(0)                            # Computa o acerto total para o batch inteiro e adiciona ao train_acc
            # Atualiza a progress bar de treino
            progress_bar_training.set_description(
                f"TREINO:\t\t\t\tLoss:\t{loss.item():.4f}, Accuracy: {acc.item() * 100:.2f}%")
        scheduler.step()    # Dá Step no "contador" que diminui o Learning Rate

        # Criação da Barra de Progresso da Validação
        progress_bar_validation = tqdm(enumerate(valid_data_loader), total=len(valid_data_loader),
                                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}] \033[94m')

        model.eval()  # Colocação do modelo em modo de validação
        with torch.no_grad():       # Validação
            for j, (inputs, labels) in progress_bar_validation:     # Loop de Validação
                inputs = inputs.to(device)                  # Move os dados do tensor para o device (cpu neste caso) para serem manipulados como inputs
                labels = labels.to(device)                  # Move os dados do tensor para o device (cpu neste caso) para serem manipulados como labels
                outputs = model(inputs)                     # Forward pass ⇾ computar os outputs nos dados de input
                loss = loss_criterion(outputs, labels)      # Computar as perdas
                valid_loss += loss.item() * inputs.size(0)  # Computar as perdas totais para o batch e adicioná-las ao valid_loss
                ret, predictions = torch.max(outputs.data, 1)                       # Encontra o valor máximo e o seu índice na dimensão 1 do tensor outputs.data
                correct_counts = predictions.eq(labels.data.view_as(predictions))   # Compara as labels previstas com as reais e determina as predições corretas
                acc = torch.mean(correct_counts.type(torch.FloatTensor))            # Converte o correct_counts para float e computa a média para obter o acerto
                valid_acc += acc.item() * inputs.size(0)                            # Computa o acerto total para o batch inteiro e adiciona ao valid_acc
                # Atualiza a progress bar de validação
                progress_bar_validation.set_description(
                    f"VALIDAÇÃO:\t\t\tLoss:\t{loss.item():.4f}, Accuracy: {acc.item() * 100:.2f}%")

        if valid_loss < best_loss:      # Guarda a perda mais baixa para saber qual a melhor época do treino
            best_loss = valid_loss
            best_epoch = epoch

        avg_train_loss = train_loss / train_data_size   # Obter a média do training loss
        avg_train_acc = train_acc / train_data_size     # Obter a média do training accuracy
        avg_valid_loss = valid_loss / valid_data_size   # Obter a média do validation loss
        avg_valid_acc = valid_acc / valid_data_size     # Obter a média do validation accuracy
        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])  # Guardar os dados no histórico de treino
        epoch_end = time.time()     # Obtenção do tempo no fim do treino

        print(Fore.LIGHTGREEN_EX + "Treino (Média):\t\tLoss:\t{:.4f}, Accuracy: {:.2f}%".format(avg_train_loss, avg_train_acc * 100))
        print(Fore.LIGHTBLUE_EX + "Validação (Média):\tLoss:\t{:.4f}, Accuracy: {:.2f}%".format(avg_valid_loss, avg_valid_acc * 100))
        print(Fore.YELLOW + "\t\t\t\t\tTempo:\t{:.4f} segundos".format(epoch_end - epoch_start))
        print(Style.RESET_ALL)
        torch.save(model, "./Fine_Tunning_History/Grapevine Leaves Classification - Fine Tunning" + '_model_' + str(epoch) + '.pt')   # Guarda o modelo
    MakeFeatureMaps(False, model, "1.png", "FT")
    return model, history, best_epoch
    pass
