import flask
from flask import Flask
from flask import request
import json
import datetime
import os
import re
import secrets

references_regex = re.compile(r"\[\d+\]")

app = Flask(__name__)


@app.route('/get_squash_doc', methods=['GET'])
def get_squash_doc():
    squash_key = request.args['id']

    queue_number = 0

    with open("../../squash-generation/squash/generated_outputs/queue/queue.txt", "r") as f:
        for i, line in enumerate(f):
            if squash_key == line.strip():
                queue_number = i + 1

    with open("../../squash-generation/squash/generated_outputs/inputs/%s/metadata.json" % squash_key, "r") as f:
        metadata = json.loads(f.read())

    if queue_number == 0:
        with open('../../squash-generation/squash/generated_outputs/final/%s.json' % squash_key, 'r') as f:
            squash_data = json.loads(f.read())
        status = None
    else:
        squash_data = None
        status = {
            "answers_extracted": os.path.exists("../../squash-generation/squash/generated_outputs/inputs/%s/input.pkl" % squash_key),
            "questions_generated": os.path.exists("../../squash-generation/squash/generated_outputs/generated_questions/%s.json" % squash_key),
            "answers_generated": os.path.exists("../../squash-generation/squash/generated_outputs/generated_answers/%s/predictions.json" % squash_key),
            "questions_filtered": False
        }

    response = flask.jsonify({
        "squash_data": squash_data,
        "queue_number": queue_number,
        "settings": metadata["settings"],
        "input_text": metadata["input_text"],
        "status": status
    })

    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/request_squash_doc', methods=['POST'])
def request_squash_doc():
    form_data = json.loads(request.data.decode('utf-8'))
    keygen = secrets.token_hex(12)

    form_data["timestamp"] = str(datetime.datetime.now())
    form_data["key"] = keygen

    # Filter out extra newline characters between paragraphs
    input_text_list = [x for x in form_data["input_text"].split("\n") if len(x.strip()) > 0]
    # Filter out input text with more than 3 paragraphs
    input_text_list = input_text_list[:3]
    # Finally, truncate paragraphs with more than 2000 characters
    form_data["input_text"] = "\n".join([x[:2000] for x in input_text_list])

    for reference in references_regex.findall(form_data["input_text"]):
        form_data["input_text"] = form_data["input_text"].replace(reference, " ")

    with open("../../squash-generation/squash/generated_outputs/queue/queue.txt", "a") as f:
        f.write("%s\n" % keygen)

    os.mkdir("../../squash-generation/squash/generated_outputs/inputs/%s" % keygen)
    os.mkdir("../../squash-generation/squash/generated_outputs/generated_answers/%s" % keygen)

    with open("../../squash-generation/squash/generated_outputs/inputs/%s/metadata.json" % keygen, "w") as f:
        f.write(json.dumps(form_data))

    response = flask.jsonify({
        "new_id": keygen
    })

    response.headers.add('Access-Control-Allow-Origin', '*')
    return response
