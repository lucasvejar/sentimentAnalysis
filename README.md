# Sentiment Analysis of Tweets

_Este proyecto se basa en realizar sentiment analysis a tweets utilizando Sickit-Learn, Keras y Textblob. El sentiment analysis se realiza a tweets obtenidos con Tweepy de mi usuario de Twitter._

## Comenzando 🚀

_El proyecto esta dividido en diferentes carpetas_
```
\----- \datasets 
   |-- \models
   |-- \scripts
   |-- \preprocessed-twitter-tweets
   |-- \test
   |-- \train
   |-- requirements.txt
   |__ .gitignore
```

Los modelos con los cuales realize las pruebas estan dentro de la carpea \models y son: modelSk1.ipynb, modelSk2.ipynb, modelKeras1.ipynb, modelTextblob.ipynb.

### Pre-requisitos 📋

_Antes de intentar correr los modelos que estan en la carpeta \models es necesario tener una cuenta en Twitter for developers, crear una aplicacion alli, y conseguir las credenciales para usar la API._
Se puede hacer facilmente siguiendo este tutorial: https://www.vozidea.com/crear-una-aplicacion-en-twitter-para-usar-la-api.
Luego de tener las credenciales, debes crear el archivo credentials.py dentro de la carpeta \models y añadir las credenciales de la siguiente manera.
```
CONSUMER_KEY = 'some_password'
CONSUMER_SECRET = 'some_password'
ACCESS_TOKEN = 'some_password'
ACCESS_TOKEN_SECRET = 'some_password'
```

### Instalación 🔧

_Es necesario tener instalado pip o pip3 para poder instalar las librerias.
Para instalar las librerias y paquetes que se utilizan hay que hacer:_

```
pip install -r requirements.txt
```
_o en el caso de tener pip3_

```
pip3 install -r requirements.txt
```

