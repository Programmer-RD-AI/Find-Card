from flask import *
from flask_restful import API

app = Flask(__name__)
app.debug = True
app.secret_key = 'secret_key'
api = API(app)
