import tensorflow as tf

# Cargar tu modelo actual
model = tf.keras.models.load_model('model.h5')

# Convertir a TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()


with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("¡Conversión exitosa! Se creó el archivo 'model.tflite'")