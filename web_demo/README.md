# Generating Question-Answer Hierarchies - Website

This is the official repository for the website and demo accompanying the ACL 2019 long paper *[Generating Question-Answer Hierarchies](https://arxiv.org/abs/1906.02622)*. The main codebase and data can be found in [martiansideofthemoon/squash-generation](https://github.com/martiansideofthemoon/squash-generation), in the `demo` branch.

To get started, place this repository as a sister folder to the repository [martiansideofthemoon/squash-generation](https://github.com/martiansideofthemoon/squash-generation). Switch to the `demo` branch using `git checkout demo`. Your folder structure should look like this,

```
squash-root/squash-generation
squash-root/squash-website
```

## SQUASH Landing Page

The landing page is a static HTML file which can be found under [`squash-landing`](squash-landing). All the code is written in [`squash-landing/index.html`](squash-landing/index.html). The website is hosted at [http://squash.cs.umass.edu/](http://squash.cs.umass.edu/). This file has been adapted from [Rowan Zeller](https://rowanzellers.com)'s landing page for [HellaSwag](https://rowanzellers.com/hellaswag/).

## SQUASH Backend

The code for the SQUASH APIs and backend is found under [`squash-backend`](squash-backend). All the code is written in [`squash-backend/app.py`](squash-backend/app.py). The code requires Python 3.6+ (for the `secrets` module) as well as the python package [Flask](https://palletsprojects.com/p/flask/). This code triggers the scripts in the `demo` branch of the main SQUASH repository. To get started,

```
cd squash-backend
export FLASK_APP=app.py
python -m flask run --host 0.0.0.0 --port 3005
```

Remove the `--host 0.0.0.0` flag if you do not want to expose the APIs publicly. Also note that you will need to restart the Flask server to reflect edits in the codebase.

Next, in a different terminal enter the `squash-generation` directory and create an empty file named `squash/generated_outputs/queue/queue.txt`.

```
touch squash/generated_outputs/queue/queue.txt
```

Finally, in five different terminals (all with `squash-generation` as the root folder launch the following scripts

```
# terminal 1
python squash/extract_answers.py

# terminal 2
python question-generation/interact.py --model_checkpoint question-generation/gpt2_corefs_question_generation --model_type gpt2

# terminal 3
python question-answering/run_squad_demo.py --bert_model question-answering/bert_large_qa_model --do_predict --do_lower_case --predict_batch_size 16 --version_2_with_negative

# terminal 4
python squash/filter.py

# terminal 5
python squash/cleanup.py
```

(For running these commands together, you might find the `tmux` command under **Production Level Deployment** useful)

## SQUASH Frontend

The SQUASH frontend has been written in [ReactJS](http://reactjs.org/). To get started, make sure you the latest `npm` and `node` installed ([reference](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm)). The dependencies for the frontend have been specified in [`squash-frontend/package.json`](squash-frontend/package.json).

To get started, first edit the [`squash-frontend/src/url.js`](squash-frontend/src/url.js) to point to the local server URL. Then, install the dependencies and run the frontend server (while the backend is running on a different process and port),

```
npm install
npm start
```

## Production Level Deployment

For a production level deployment, you should not use developmental servers. For the backend server, we use [waitress](https://docs.pylonsproject.org/projects/waitress/en/stable/). To run the server use,

```
cd squash-backend
python waitress_server.py
```

For the frontend server, first create a static website and then [serve](https://www.npmjs.com/package/serve) it. See [here](https://facebook.github.io/create-react-app/docs/deployment) for other options and more details.

```
cd squash-frontend
npm run build
npx serve -s build -l 3000
```

You might find this all-in-one `tmux` command useful.

```
tmux new -s squash \
    "cd squash-website/squash-backend ; python waitress_server.py ; read" \; \
    new-window "cd squash-website/squash-frontend ; npx serve -s build -l 3000 ; read" \; \
    new-window "cd squash-generation ; python squash/extract_answers.py ; read" \; \
    new-window "cd squash-generation ; export CUDA_VISIBLE_DEVICES=0 ; python question-generation/interact.py --model_checkpoint question-generation/gpt2_corefs_question_generation --model_type gpt2 ; read" \; \
    new-window "cd squash-generation ; export CUDA_VISIBLE_DEVICES=0 ; python question-answering/run_squad_demo.py --bert_model question-answering/bert_large_qa_model --do_predict --do_lower_case --predict_batch_size 16 --version_2_with_negative ; read" \; \
    new-window "cd squash-generation ; python squash/filter.py ; read" \; \
    new-window "cd squash-generation ; python squash/cleanup.py ; read" \; \
    detach \;
```

## Citation

If you find this website demo useful, please cite us.

```
@inproceedings{squash2019,
Author = {Kalpesh Krishna and Mohit Iyyer},
Booktitle = {Association for Computational Linguistics,
Year = "2019",
Title = {Generating Question-Answer Hierarchies}
}
```
