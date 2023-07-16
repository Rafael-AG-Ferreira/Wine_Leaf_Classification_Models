import torch


def ComputeTestSetAccuracy(model, loss_criterion, test_data_loader, test_data_size):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_acc = 0.0      # Inicialização da accuracy
    test_loss = 0.0     # Inicialização da loss

    with torch.no_grad():       # Validação
        model.eval()            # Colocação do modelo em modo de validação
        for j, (inputs, labels) in enumerate(test_data_loader):     # Loop de Validação
            inputs = inputs.to(device)                              # Move os dados do tensor para o device (cpu neste caso) para serem manipulados como inputs
            labels = labels.to(device)                              # Move os dados do tensor para o device (cpu neste caso) para serem manipulados como labels
            outputs = model(inputs)                                 # Forward pass ⇾ computar os outputs nos dados de input
            loss = loss_criterion(outputs, labels)                              # Computar as perdas
            test_loss += loss.item() * inputs.size(0)                           # Computar as perdas totais para o batch e adicioná-las ao valid_loss
            ret, predictions = torch.max(outputs.data, 1)                       # Encontra o valor máximo e o seu índice na dimensão 1 do tensor outputs.data
            correct_counts = predictions.eq(labels.data.view_as(predictions))   # Compara as labels previstas com as reais e determina as predições corretas
            acc = torch.mean(correct_counts.type(torch.FloatTensor))            # Converte o correct_counts para float e computa a média para obter o acerto
            test_acc += acc.item() * inputs.size(0)                             # Computa o acerto total para o batch inteiro e adiciona ao test_acc

            print("A testar o batch nº: {:03d}, Test: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))

    avg_test_loss = test_loss / test_data_size      # Obter a média do test loss
    avg_test_acc = test_acc / test_data_size        # Obter a média do test accuracy
    print("Accuracy : " + str(avg_test_acc))        # Imprimir a média do test accuracy
    print("Loss : " + str(avg_test_loss))           # Imprimir a média do test loss
    pass
