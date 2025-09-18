from flask import Blueprint, render_template

main = Blueprint('main', __name__)

@main.route('/')
def home():
    return render_template("index.html", active_page="home")

@main.route('/how-it-works')
def howitworks():
    return render_template("howitworks.html", active_page="howitworks")

@main.route('/upload')
def upload():
    return render_template("upload.html", active_page="upload")

@main.route('/result')
def result():
    return render_template("result.html", active_page="result")

@main.route('/condition')
def condition():
    return render_template("condition.html", active_page="condition")
