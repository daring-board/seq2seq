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
    lines = data['history'], data['uttence']
    res, history = app.config['engine'].response(lines)
    print(history)
    print(res)
    return jsonify({'response': res, 'history': history})
 
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)