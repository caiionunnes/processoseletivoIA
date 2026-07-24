import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import os

# ---------------------------------------------------------------------------
# Projeto 1 — Otimização do Modelo (MNIST)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo treinado em "model.h5"
#   2. Converter para TensorFlow Lite usando tf.lite.TFLiteConverter
#   3. Aplicar uma técnica de otimização (ex: Dynamic Range Quantization,
#      via converter.optimizations = [tf.lite.Optimize.DEFAULT])
#   4. Salvar o resultado como "model.tflite"
# ---------------------------------------------------------------------------

import tensorflow as tf

# 1. Carregamento do model.h5 treinado
print("Carregando o modelo original (model.h5)...")
model = tf.keras.models.load_model('model.h5')

# 2. Configuração da Conversão para TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 3. Aplicação da técnica de otimização (Dynamic Range Quantization)
# Isso reduz o tamanho dos pesos de 32-bit (float) para 8-bit (int), otimizando recursos.
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Realiza a conversão
print("Otimizando e convertendo o modelo para formato Edge...")
tflite_model = converter.convert()

# 4. Salvamento do modelo otimizado
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Sucesso! Modelo otimizado salvo como 'model.tflite'.")
