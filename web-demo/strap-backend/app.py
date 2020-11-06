import flask
from flask import Flask
from flask import request
import json
import datetime
import os
import re
import secrets

app = Flask(__name__)

with open("../config.json", "r") as f:
    configuration = json.loads(f.read())
    OUTPUT_DIR = configuration["output_dir"]


@app.route('/get_strap_doc', methods=['GET'])
def get_strap_doc():
    strap_key = request.args['id']

    queue_number = 0

    with open(OUTPUT_DIR + "/generated_outputs/queue/queue.txt", "r") as f:
        for i, line in enumerate(f):
            if strap_key == line.strip():
                queue_number = i + 1

    with open(OUTPUT_DIR + "/generated_outputs/inputs/%s/metadata.json" % strap_key, "r") as f:
        metadata = json.loads(f.read())

    if queue_number == 0:
        with open(OUTPUT_DIR + "/generated_outputs/final/%s.json" % strap_key, 'r') as f:
            strap_data = json.loads(f.read())
        status = None
    else:
        strap_data = None
        status = "processing input..."

    response = flask.jsonify({
        "output_data": strap_data,
        "queue_number": queue_number,
        "settings": metadata["settings"],
        "input_text": metadata["input_text"],
        "status": status,
        "target_style": metadata["target_style"],
    })

    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/request_strap_doc', methods=['POST'])
def request_strap_doc():
    form_data = json.loads(request.data.decode('utf-8'))
    keygen = secrets.token_hex(12)

    form_data["timestamp"] = str(datetime.datetime.now())
    form_data["key"] = keygen
    form_data["input_text"] = " ".join(form_data["input_text"].split())

    with open(OUTPUT_DIR + "/generated_outputs/queue/queue.txt", "a") as f:
        f.write("%s\n" % keygen)

    os.mkdir(OUTPUT_DIR + "/generated_outputs/inputs/%s" % keygen)

    with open(OUTPUT_DIR + "/generated_outputs/inputs/%s/metadata.json" % keygen, "w") as f:
        f.write(json.dumps(form_data))

    with open(OUTPUT_DIR + "/generated_outputs/inputs/%s/written.txt" % keygen, "w") as f:
        f.write("True")

    response = flask.jsonify({
        "new_id": keygen
    })

    response.headers.add('Access-Control-Allow-Origin', '*')
    return response
