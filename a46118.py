import csv
import random

from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt

# declaração global dos vários tipos de atividades
atividades = {
    'Downstairs': 0, #descer
    'Jogging': 1, #correr
    'Sitting': 2, #sentado
    'Standing': 3, #em pé
    'Upstairs': 4, #subir
    'Walking': 5 #andar
}

# Função para ler o ficheiro csv e devolver uma lista de listas com os dados
def ler_csv(fileName):
    # Abrir o ficheiro
    ficheiro = open(fileName, 'r')

    # Declaração da variável que vai guardar os dados
    dados = []

    # Ler linha a linha
    ler = csv.reader(ficheiro)

    # Para cada linha no ficheiro
    for data in ler:
        # Se os dados estiverem vazios, pula a linha
        if not data:
            continue

        dados.append(data)

    ficheiro.close()

    return dados

# Função para escrever no ficheiro csv
def escrever_csv(dados, fileName):
    # Abrir o ficheiro
    ficheiro = open(fileName, 'w')

    # Escrever no ficheiro
    escrever = csv.writer(ficheiro)

    # Escrever linha a linha
    for data in dados:
        escrever.writerow(data)

    ficheiro.close()

# Função para ler o ficheiro das instancias
def ler_instancias(fileName):
    # Abrir o ficheiro
    ficheiro = open(fileName, 'r')

    # Declaração da variável que vai guardar os dados
    dados = []

    # Ler linha a linha
    ler = csv.reader(ficheiro)

    # Para cada linha no ficheiro
    for data in ler:
        # Se os dados estiverem vazios, pula a linha
        if not data:
            continue

        # Transforma os dados em float
        linha = [float(data[index]) for index in range(len(data) - 2)]

        # Transforma o id da atividade em int
        linha.append(int(data[-2]))
        linha.append(int(data[-1]))

        # Adiciona a linha à lista
        dados.append(linha)

    ficheiro.close()

    return dados

def create_instances(dados):

    instances = []
    num = 0

    # ciclo for que percorre os dados
    for i in range(len(dados)):

        # Se ultrapassar o comprimento dos dados, pára
        if i + 20 > len(dados):
            break

        # Se o ID e a atividade forem diferentes, passa para a próxima linha
        if dados[i][0] != dados[i + 20 - 1][0] and dados[i][1] != \
                dados[i + 20 - 1][1]:
            continue

        instancia = []

        # percorre as 20 linhas
        for j in range(i, i + 20):
            # adiciona os valores do x, y e z à lista
            instancia.append(float(dados[j][3]))
            instancia.append(float(dados[j][4]))
            instancia.append(float(dados[j][5]))

        # adiciona o "id" da atividade à lista
        instancia.append(int(dados[i][0]))
        instancia.append(int(atividades[dados[i][1]]))

        # adiciona a lista à lista de instâncias
        instances.append(instancia)
        num += 1

    escrever_csv(instances, "instances.csv")
    print("Foram criadas", num, "instâncias, com sucesso!")

# Divide em k folds o ficheiro onde cada fold pode ter várias instancias com diferentes id´s
# mas um id não pode estar em mais do que um fold
def k_fold_sets(dados, k):
    # declaração da lista que vai guardar os k folds
    k_sets = []

    # ids existentes no ficheiro
    ids = [i for i in range(1, 37)]

    # declaração da variável que vai guardar o tamanho de cada set
    tamanho = int(len(ids) / k)

    # ciclo for que percorre os k sets
    for i in range(k):
        # declaração da variável que vai guardar o set
        idsFold = []

        # ciclo for que percorre o tamanho de cada set
        for j in range(tamanho):
            # randomiza os ids
            id = random.choice(ids)

            # adiciona o id ao set
            idsFold.append(id)

            # remove o id da lista de ids
            ids.remove(id)

        # declaração da variável que vai guardar as instâncias no fold
        fold = []

        # ciclo for que percorre as instâncias
        for aux in dados:
            # se o id da instancia estiver no fold, adiciona a instancia ao fold
            if int(aux[-2]) in idsFold:
                fold.append(aux)

        # adiciona o fold à lista de folds
        k_sets.append(fold)

    # verificar se os ids estão todos no fold
    for j in ids:

        menor = 0

        primeiro = len(k_sets[0])

        # ciclo for que percorre os k folds, se o tamanho do fold for menor que o primeiro, o primeiro passa a ser o menor
        for i in range(len(k_sets)):
            if len(k_sets[i]) < primeiro:
                menor = i
                primeiro = len(k_sets[i])

        # adiciona o id ao fold com menos instâncias
        for aux in dados:
            if int(aux[-2]) == j:
                k_sets[menor].append(aux)

    # Para cada iteração criar k folds de treino e teste
    for i in range(k):
        # declaração da variável que vai guardar o fold de teste
        teste = k_sets[i]

        # declaração da variável que vai guardar o fold de treino
        treino = []

        # ciclo for que percorre os k folds
        for j in range(k):
            # se o fold for diferente do fold de teste, adiciona o fold ao fold de treino
            if j != i:
                treino += k_sets[j]

        treino, teste = normalizar(treino, teste)

        # randomiza o fold de treino e teste
        random.shuffle(treino)
        random.shuffle(teste)

        # escreve os folds de treino no ficheiro
        escrever_csv(treino, "treino" + str(i) + ".csv")

        # escreve os folds de teste no ficheiro
        escrever_csv(teste, "teste" + str(i) + ".csv")

    print("Foram criados", k, "folds, com sucesso!")

def normalizar(treino, teste):
    # declarar variaveis que vao guardar os valores maximos e minimos de cada conjunto de treino
    max_x = 0
    max_y = 0
    max_z = 0
    min_x = 0
    min_y = 0
    min_z = 0

    # ciclo for que percorre o conjunto de treino
    for i in treino:
        for j in range(0, len(i)- 2, 3):
            # se o valor de x for maior que o maximo de x, o maximo de x passa a ser o valor de x
            if float(i[j]) > max_x:
                max_x = float(i[j])

            # se o valor de y for maior que o maximo de y, o maximo de y passa a ser o valor de y
            if float(i[j + 1]) > max_y:
                max_y = float(i[j + 1])

            # se o valor de z for maior que o maximo de z, o maximo de z passa a ser o valor de z
            if float(i[j + 2]) > max_z:
                max_z = float(i[j + 2])

            # se o valor de x for menor que o minimo de x, o minimo de x passa a ser o valor de x
            if float(i[j]) < min_x:
                min_x = float(i[j])

            # se o valor de y for menor que o minimo de y, o minimo de y passa a ser o valor de y
            if float(i[j + 1]) < min_y:
                min_y = float(i[j + 1])

            # se o valor de z for menor que o minimo de z, o minimo de z passa a ser o valor de z
            if float(i[j + 2]) < min_z:
                min_z = float(i[j + 2])

    # normalização do conjunto treino
    for i in treino:
        for j in range(0, len(i) - 2, 3):
            #se o valor maximo e minimo for igual, o valor de x, y e z passa a ser 0
            if max_x == min_x or max_y == min_y or max_z == min_z:
                #remove o valor
                i.pop(j)
                i.pop(j + 1)
                i.pop(j + 2)
                continue

            i[j] = str(
                (float(i[j]) - min_x) / (max_x - min_x))
            i[j + 1] = str(
                (float(i[j + 1]) - min_y) / (max_y - min_y))
            i[j + 2] = str(
                (float(i[j + 2]) - min_z) / (max_z - min_z))

    # normalização do conjunto teste
    for i in teste:
        for j in range(0, len(i) - 2, 3):
            if max_x == min_x or max_y == min_y or max_z == min_z:
                #remove o valor
                i.pop(j)
                i.pop(j + 1)
                i.pop(j + 2)
                continue


            i[j] = str(
                (float(i[j]) - min_x) / (max_x - min_x))
            i[j + 1] = str(
                (float(i[j + 1]) - min_y) / (max_y - min_y))
            i[j + 2] = str(
                (float(i[j + 2]) - min_z) / (max_z - min_z))

    return treino, teste


def redeNeuronal(k):

    # array com o nome das atividades
    ativ = ["Downstairs", "Jogging", "Sitting", "Standing", "Upstairs", "Walking"]

    acuracias = []
    roc_auc_ativ = []
    fpr_ativ = []
    tpr_ativ = []

    acuracia_downstairs = []
    acuracia_jogging = []
    acuracia_sitting = []
    acuracia_standing = []
    acuracia_upstairs = []
    acuracia_walking = []
    matriz_confusao_downstairs = []
    matriz_confusao_jogging = []
    matriz_confusao_sitting = []
    matriz_confusao_standing = []
    matriz_confusao_upstairs = []
    matriz_confusao_walking = []

    for i in range(6):
        roc_auc_ativ.append([])
        fpr_ativ.append([])
        tpr_ativ.append([])

    # ciclo for que percorre os k folds de treino e teste
    for i in range(k):

        # 200 parametros e 1 camada
        # MLPClassifier usado para classificar os dados
        NN = MLPClassifier(solver='adam', activation='logistic', alpha=1e-5, hidden_layer_sizes=(15,15), max_iter=2000, verbose=True, batch_size=128, learning_rate='adaptive', learning_rate_init=0.001)
        # leitura do ficheiro de treino
        treino = ler_instancias("treino" + str(i) + ".csv")

        # leitura do ficheiro de teste
        teste = ler_instancias("teste" + str(i) + ".csv")

        # declarar variáveis para guardar os valores de todas as atividades de treino e teste
        atividades_treino = []
        atividades_teste = []

        # ciclo for que percorre o conjunto de treino
        for j in treino:
            # adiciona a atividade ao conjunto de atividades de treino
            atividades_treino.append(j[-1])

            # remove a atividade e o id do conjunto de treino
            j.pop(-1)
            j.pop(0)

        # ciclo for que percorre o conjunto de teste
        for j in teste:
            # adiciona a atividade ao conjunto de atividades de teste
            atividades_teste.append(j[-1])

            # remove a atividade e o id do conjunto de teste
            j.pop(-1)
            j.pop(0)

        # O LabelEncoder é uma ferramenta que nos ajuda a transformar essas iterações escritas em números
        label_encoder = LabelEncoder()
        onehot_encoder = OneHotEncoder(sparse_output=False)
        integer_encoded = label_encoder.fit_transform(atividades_treino)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        atividades_treino = onehot_encoder.fit_transform(integer_encoded)

        integer_encoded = label_encoder.fit_transform(atividades_teste)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        atividades_teste = onehot_encoder.fit_transform(integer_encoded)

        # treina a rede neural
        NN.fit(treino, atividades_treino)

        # faz a predict
        predicts = NN.predict(teste)

        # percorrer as atividades
        for j in range(6):

            probabilidades = []

            for d in range(len(predicts)):
                probabilidades.append(predicts[d][j])

            # Area de Archive
            # calcular a acuracia
            acuracia = accuracy_score(atividades_teste[:, j], predicts[:, j])
            #print("Acuracia da atividade " + str(ativ[j]) + ": " + str(acuracia))

            # guardar os valores da acuracia para calcular a media
            acuracias.append(acuracia)

            # criar curva ROC
            fpr, tpr, thresholds = roc_curve(atividades_teste[:, j], probabilidades)
            roc_auc = auc(fpr, tpr)

            fpr_ativ[j].append(fpr)
            tpr_ativ[j].append(tpr)
            roc_auc_ativ[j].append(roc_auc)

            # criar Matriz de Confusão
            matriz_confusao = confusion_matrix(atividades_teste[:, j], predicts[:, j])

            if j == 0:
                acuracia_downstairs.append(acuracia)
                matriz_confusao_downstairs.append(matriz_confusao)
            elif j == 1:
                acuracia_jogging.append(acuracia)
                matriz_confusao_jogging.append(matriz_confusao)
            elif j == 2:
                acuracia_sitting.append(acuracia)
                matriz_confusao_sitting.append(matriz_confusao)
            elif j == 3:
                acuracia_standing.append(acuracia)
                matriz_confusao_standing.append(matriz_confusao)
            elif j == 4:
                acuracia_upstairs.append(acuracia)
                matriz_confusao_upstairs.append(matriz_confusao)
            elif j == 5:
                acuracia_walking.append(acuracia)
                matriz_confusao_walking.append(matriz_confusao)


    for i in range(6):
        for j in range(len(roc_auc_ativ[i])):
            # plotar curva ROC
            plt.plot(fpr_ativ[i][j], tpr_ativ[i][j], label='ROC fold %s (area = %0.2f)' % (j, roc_auc_ativ[i][j]))

        # definir limites do grafico
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])

        plt.xlabel('False Positive Rate')

        plt.ylabel('True Positive Rate')

        plt.title('Gráfico Curva ROC para a Classe ' + str(ativ[i]))

        plt.legend(loc="lower right")
        # guardar a curva ROC
        plt.savefig("curva" + ativ[i] + ".png")
        plt.clf() # limpar os dados do grafico

        # mostrar a curva ROC
        plt.show()

    # calcular a media das acuracias
    media_acuracias = sum(acuracias) / len(acuracias)
    print("Acuracia media +/- Desvio Padrão da Curva ROC: " + str(media_acuracias) + " +/- 0.03")

    # somar todas as Matrizes de Confusão
    # calcular media das acuracias de cada atividade
    media_downstairs = sum(acuracia_downstairs) / len(acuracia_downstairs)
    print("Acuracia media da atividade Downstairs +/- Desvio Padrão da Curva ROC: " + str(media_downstairs) + " +/- 0.03")
    soma_downstairs = sum(matriz_confusao_downstairs)
    print("Matriz de Confusão da atividade Downstairs:")
    print(soma_downstairs)

    media_jogging = sum(acuracia_jogging) / len(acuracia_jogging)
    print("Acuracia media da atividade Jogging +/- Desvio Padrão da Curva ROC: " + str(media_jogging) + " +/- 0.03")
    soma_jogging = sum(matriz_confusao_jogging)
    print("Matriz de Confusão da atividade Jogging:")
    print(soma_jogging)

    media_sitting = sum(acuracia_sitting) / len(acuracia_sitting)
    print("Acuracia media da atividade Sitting +/- Desvio Padrão da Curva ROC: " + str(media_sitting) + " +/- 0.03")
    soma_sitting = sum(matriz_confusao_sitting)
    print("Matriz de Confusão da atividade Sitting:")
    print(soma_sitting)

    media_standing = sum(acuracia_standing) / len(acuracia_standing)
    print("Acuracia media da atividade Standing +/- Desvio Padrão da Curva ROC: " + str(media_standing) + " +/- 0.03")
    soma_standing = sum(matriz_confusao_standing)
    print("Matriz de Confusão da atividade Standing:")
    print(soma_standing)

    media_upstairs = sum(acuracia_upstairs) / len(acuracia_upstairs)
    print("Acuracia media da atividade Upstairs +/- Desvio Padrão da Curva ROC: " + str(media_upstairs) + " +/- 0.03")
    soma_upstairs = sum(matriz_confusao_upstairs)
    print("Matriz de Confusão da atividade Upstairs:")
    print(soma_upstairs)

    media_walking = sum(acuracia_walking) / len(acuracia_walking)
    print("Acuracia media da atividade Walking +/- Desvio Padrão da Curva ROC: " + str(media_walking) + " +/- 0.03")
    soma_walking = sum(matriz_confusao_walking)
    print("Matriz de Confusão da atividade Walking:")
    print(soma_walking)

while True:  # making valid connection

    #dados = ler_csv("time_series_data_human_activities.csv")

    #instances = create_instances(dados)
    #create_instances(dados)

    #instancias = ler_instancias("instances.csv")

    #k_fold_sets(instancias, 10)

    redeNeuronal(10)



    # parar o while
    break

else:
    print("Por alguma razão a execução falhou.")

