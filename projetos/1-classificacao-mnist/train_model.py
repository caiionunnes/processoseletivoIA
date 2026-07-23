import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------------------
# Projeto 1 — Classificação MNIST
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o dataset MNIST via tf.keras.datasets.mnist
#   2. Normalizar as imagens para [0, 1] e ajustar o shape para (28, 28, 1)
#   3. Separar um conjunto de validação (ex: validation_split ou split manual)
#   4. Construir uma CNN com 3-4 blocos Conv2D + BatchNormalization + MaxPooling2D,
#      seguida de Dropout antes da camada de saída (10 classes, softmax)
#   5. Treinar com EarlyStopping monitorando a perda de validação
#   6. Exibir a acurácia de validação final no terminal
#   7. Salvar o modelo treinado como "model.h5"
# ---------------------------------------------------------------------------

# 1. Carregar o dataset MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 2. Normalizar as imagens para [0, 1] e ajustar o shape para (28, 28, 1)
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# 4. Construir uma CNN com 3 blocos
model = keras.Sequential([
    # Bloco 1
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Bloco 2
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Bloco 3
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Achatamento e Dropout
    layers.Flatten(),
    layers.Dropout(0.5), # Dropout antes da camada de saída para evitar overfitting
    
    # Camada de saída (10 classes, softmax)
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Treinar com EarlyStopping monitorando a perda de validação
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3, 
    restore_best_weights=True 
)

print("Iniciando o treinamento da CNN...")

# 3. Separar um conjunto de validação (validation_split=0.2)
history = model.fit(
    x_train, y_train,
    epochs=15, 
    batch_size=64,
    validation_split=0.2, 
    callbacks=[early_stopping]
)

# 6. Exibir a acurácia de validação final no terminal
val_acc = history.history['val_accuracy'][-1]
print(f"\nTreinamento concluído! Acurácia de validação final: {val_acc:.2%}")

# 7. Salvar o modelo treinado como "model.h5"
model.save("model.h5")
print("Modelo salvo com sucesso como 'model.h5'!")
