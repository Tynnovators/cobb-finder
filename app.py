from flask import Flask, request, flash, redirect, url_for, send_file, render_template

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/demo')
def demo():
    return render_template('demo.html')


@app.route('/results')
def results():
    return render_template('results.html')


if __name__ == '__main__':
    app.run()
