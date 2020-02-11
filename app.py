from flask import Flask, request, jsonify
from util import ChatEngine
from flask_cors import CORS
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['engine'] = ChatEngine()
CORS(app)

@app.route('/')
def hello():
    hello = "Hello world"
    return hello

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    print(data)
    lines = data['uttence1'], data['uttence2']
    res = app.config['engine'].response(lines)
    res = res.replace('▁', '')
    print(res)
    return jsonify(res)
 
if __name__ == "__main__":
    app.run(debug=True)