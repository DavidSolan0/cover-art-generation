from flask import Flask, render_template
from flask_bootstrap import Bootstrap
from flask import abort, request

app = Flask(__name__)
bootstrap = Bootstrap(app)


# app.template_folder = '/var/www/cover/covert-art-generation/'

@app.before_request
def limit_remote_addr():
    if request.remote_addr != '37.223.179.210':
        abort(403)  # Forbidden

@app.route('/cover')
def index():
    return render_template('index.html', bootstrap=bootstrap)


@app.route('/')
def hello():
    return "¡Hola, mundo!"


@app.route('/upload', methods=['POST'])
def upload():
    # Obtener los archivos enviados desde el formulario
    song = request.files['song']
    # image = request.files['image']
    # Guardar los archivos en una ubicación específica
    song.save('cancion.mp3')
    # image.save('imagen.png')

    # Realizar acciones adicionales con los archivos si es necesario
    return 'Archivos subidos con éxito'

"""
@app.route('/', methods=['POST'])
def upload():
    song_file = request.files['song']
    image_file = request.files['image']
    # Guardar la canción en la carpeta 'static'
    song_file.save(os.path.join('static', song_file.filename))
    # Guardar la imagen en la carpeta 'static/images'
    image_path = os.path.join('static', 'images', image_file.filename)
    image_file.save(image_path)
    return 'Carga exitosa'
"""

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8080')

"""

from flask import Flask

app = Flask(__name__)

@app.route('/run')
def hello():
    return "¡Hola, mundo como estas!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000')

"""