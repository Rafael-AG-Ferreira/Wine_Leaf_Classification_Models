import splitfolders
import random


def dataSetSplit(trainRatio, validationRatio, testRatio):   # Função para fazer o Dataset Split
    randSeed = random.randint(1, 9999)  # Obtenção de uma seed aleatória
    splitfolders.ratio('../../DataV3', output="Splitted_Dataset", seed=randSeed, ratio=(trainRatio, validationRatio, testRatio))
    print("As divisões do dataset estão concluídas")
    pass
