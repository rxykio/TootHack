from flask import Blueprint, render_template

def home():
    return render_template("index.html")

def howitworks():
    return render_template("howitworks.html")

def upload():
    return render_template("upload.html")

def result():
    return render_template("result.html")

def condition():
    return render_template("condition.html")