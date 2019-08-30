from flask import Flask
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

from app import views
from app.helpers import get_page_display_name, get_page_url_name

app.jinja_env.globals.update(get_page_display_name=get_page_display_name)
app.jinja_env.globals.update(get_page_url_name=get_page_url_name)