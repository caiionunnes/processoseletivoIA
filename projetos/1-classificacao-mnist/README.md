

## 📝 Relatório do Candidato

👤 **Nome Completo:Caio de Souza Nunes**

### 1️⃣ Resumo da Arquitetura do Modelo

O modelo foi construído utilizando uma Rede Neural Convolucional (CNN) sequencial focada em eficiência para aplicações em Edge AI. A arquitetura é composta por 3 blocos convolucionais, onde cada bloco contém uma camada Conv2D para extração de características, seguida de BatchNormalization para estabilizar e acelerar o treinamento, e MaxPooling2D para a redução de dimensionalidade. Após a extração, os dados são achatados (Flatten) e passam por uma camada densa oculta. Antes da camada de saída (softmax com 10 classes), foi implementada uma camada de Dropout atuando como regularizador para mitigar o overfitting. O treinamento utilizou divisão explícita de dados para validação e a técnica de EarlyStopping, monitorando a perda de validação (val_loss) para interromper o processamento no ponto ótimo de generalização.

### 2️⃣ Bibliotecas Utilizadas

TensorFlow / Keras (v2.x): Para construção da arquitetura, treinamento, salvamento (.h5) e conversão para o interpretador leve (.tflite).

NumPy: Para manipulação vetorial dos arrays e pré-processamento das imagens de teste.

OS (Built-in Python): Para o gerenciamento de caminhos relativos e absolutos no carregamento do modelo.

### 3️⃣ Técnica de Otimização do Modelo

Foi aplicada a técnica de Quantização de Faixa Dinâmica (Dynamic Range Quantization) fornecida nativamente pelo TensorFlow Lite (tf.lite.Optimize.DEFAULT). Essa estratégia reduz o espaço de armazenamento e otimiza a execução ao converter os pesos estruturais do modelo de ponto flutuante de 32 bits (float32) para inteiros de 8 bits (int8). As ativações mantêm-se em ponto flutuante durante a inferência, equilibrando uma excelente taxa de compressão com uma perda quase nula de acurácia preditiva.

### 4️⃣ Resultados Obtidos

Acurácia de validação final: 98.86%

Tamanho do modelo original (model.h5): [736] KB

Tamanho do modelo otimizado (model.tflite): [67] KB

### 5️⃣ Comentários Adicionais (Opcional)

Durante a fase de testes de inferência no ambiente Windows, deparei-me com uma limitação conhecida do motor em C++ utilizado pelo tf.lite.Interpreter. O sistema gerava um erro de leitura (ValueError: Could not open) ao tentar acessar o arquivo .tflite caso houvesse caracteres especiais ou acentos na string do caminho do diretório. O problema foi solucionado transferindo o diretório de execução para uma raiz livre de caracteres especiais. Outra dificuldade foi em achar uma versão de Python compatível com o TensorFlow, mesmo com uma versão superior a 3.9 (3.14), não funcionava e tive que usar outra versão inferior do Python (3.12).

### 6️⃣ Exemplo de Inferência

Rodando inferencia em 5 amostras usando model.tflite:

Amostra 1: predito=7 | real=7
Amostra 2: predito=2 | real=2
Amostra 3: predito=1 | real=1
Amostra 4: predito=0 | real=0
Amostra 5: predito=4 | real=4

Como podemos ver o modelo otimizado apresentou 100% de acerto nas amostras de validação isoladas, comprovando que a quantização agressiva para precisão de 8 bits atendeu aos requisitos do projeto sem deteriorar o poder de classificação da rede.

