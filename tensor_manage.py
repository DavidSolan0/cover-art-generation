import tensorflow as tf
from datetime import datetime

# Directorio donde se guardarán los registros de TensorFlow
logdir = '/var/www/cover/cover-art-generation/logs'

# Crear un objeto de escritura de resúmenes
summary_writer = tf.summary.create_file_writer(logdir)

# Habilitar la ejecución rápida (eager execution)
tf.config.run_functions_eagerly(True)

# Iniciar el servidor de TensorBoard
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# Cargar el modelo o gráfico computacional de TensorFlow
# Aquí deberás agregar tu propio código para construir y entrenar tu modelo

# Entrenar el modelo y agregar resúmenes
model.fit(x_train, y_train, epochs=num_epochs, callbacks=[tensorboard_callback])

# Cerrar el objeto de escritura de resúmenes
summary_writer.close()

"""import tensorflow as tf
from tensorflow.python.summary import summary as tf_summary
from tensorflow.python.ops import gen_summary_ops
from tensorflow.core.protobuf import config_pb2

# Directorio donde se guardarán los registros de TensorFlow
logdir = '/var/www/cover/cover-art-generation/logs'

# Crear un objeto de configuración para TensorBoard
config = config_pb2.ConfigProto()
config.gpu_options.allow_growth = True

# Iniciar el servidor de TensorBoard
server = tf_summary.FileWriter(logdir)#, config=config)

# Cargar el modelo o gráfico computacional de TensorFlow
# Aquí deberás agregar tu propio código para construir y entrenar tu modelo

# Iniciar sesión de TensorFlow
with tf.Session() as sess:
    # Inicializar las variables
    sess.run(tf.global_variables_initializer())

    # Crear un resumen del gráfico computacional
    tf_summary.FileWriter.add_graph(server, sess.graph)

    # Entrenar el modelo y agregar resúmenes
    for step in range(num_steps):
        # Ejecutar el paso de entrenamiento
        sess.run(train_step)

        # Agregar resúmenes de TensorFlow
        summary = sess.run(merged_summary_op)
        server.add_summary(summary, global_step=step)

# Cerrar el servidor de TensorBoard
server.close()
"""