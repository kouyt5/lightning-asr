from flask import Flask

app = Flask(__name__)


@app.route('/')
def hello():
    return "hello2"


if __name__ == '__main__':
    app.run(port=8002, debug=True)