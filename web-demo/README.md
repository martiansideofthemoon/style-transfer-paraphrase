# Reformulating Unsupervised Style Transfer as Paraphrase Generation - Website

This is the official repository for the website and demo accompanying the EMNLP 2020 long paper *[Reformulating Unsupervised Style Transfer as Paraphrase Generation](https://arxiv.org/abs/2010.05700)*. This code is heavily based off a previous project, [martiansideofthemoon/squash-website](https://github.com/martiansideofthemoon/squash-website).


## STRAP Landing Page

The landing page is a static HTML file which can be found under [`strap-landing`](strap-landing). All the code is written in [`strap-landing/index.html`](strap-landing/index.html). The website is hosted at [http://style.cs.umass.edu/](http://style.cs.umass.edu/). This file has been adapted from [Rowan Zeller](https://rowanzellers.com)'s landing page for [HellaSwag](https://rowanzellers.com/hellaswag/).

## STRAP Backend

The code for the STRAP APIs and backend is found under [`strap-backend`](strap-backend). All the code is written in [`strap-backend/app.py`](strap-backend/app.py). The code requires Python 3.6+ (for the `secrets` module) as well as the python package [Flask](https://palletsprojects.com/p/flask/). This code triggers the scripts in the `demo` branch of the main strap repository. To get started,

```
cd strap-backend
export FLASK_APP=app.py
python -m flask run --host 0.0.0.0 --port 3005
```

Remove the `--host 0.0.0.0` flag if you do not want to expose the APIs publicly. Also note that you will need to restart the Flask server to reflect edits in the codebase.

Next, in a different terminal enter the `strap-generation` directory and create an empty file named `strap/generated_outputs/queue/queue.txt`.

```
touch strap/generated_outputs/queue/queue.txt
```

Finally, in five different terminals (all with `strap-generation` as the root folder launch the following scripts

```
# terminal 1
python strap/extract_answers.py

# terminal 2
python question-generation/interact.py --model_checkpoint question-generation/gpt2_corefs_question_generation --model_type gpt2

# terminal 3
python question-answering/run_squad_demo.py --bert_model question-answering/bert_large_qa_model --do_predict --do_lower_case --predict_batch_size 16 --version_2_with_negative

# terminal 4
python strap/filter.py

# terminal 5
python strap/cleanup.py
```

(For running these commands together, you might find the `tmux` command under **Production Level Deployment** useful)

## STRAP Frontend

The STRAP frontend has been written in [ReactJS](http://reactjs.org/). To get started, make sure you the latest `npm` and `node` installed ([reference](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm)). The dependencies for the frontend have been specified in [`strap-frontend/package.json`](strap-frontend/package.json).

To get started, first edit the [`strap-frontend/src/url.js`](strap-frontend/src/url.js) to point to the local server URL. Then, install the dependencies and run the frontend server (while the backend is running on a different process and port),

```
npm install
npm start
```

## Production Level Deployment

For a production level deployment, you should not use developmental servers. For the backend server, we use [waitress](https://docs.pylonsproject.org/projects/waitress/en/stable/). To run the server use,

```
cd strap-backend
python waitress_server.py
```

For the frontend server, first create a static website and then [serve](https://www.npmjs.com/package/serve) it. See [here](https://facebook.github.io/create-react-app/docs/deployment) for other options and more details.

```
cd strap-frontend
npm run build
npx serve -s build -l 3000
```

You might find this all-in-one `tmux` command useful.

```
tmux new -s strap \
    "cd strap-website/strap-backend ; python waitress_server.py ; read" \; \
    new-window "cd strap-website/strap-frontend ; npx serve -s build -l 3000 ; read" \; \
    new-window "cd strap-generation ; python strap/extract_answers.py ; read" \; \
    new-window "cd strap-generation ; export CUDA_VISIBLE_DEVICES=0 ; python question-generation/interact.py --model_checkpoint question-generation/gpt2_corefs_question_generation --model_type gpt2 ; read" \; \
    new-window "cd strap-generation ; export CUDA_VISIBLE_DEVICES=0 ; python question-answering/run_squad_demo.py --bert_model question-answering/bert_large_qa_model --do_predict --do_lower_case --predict_batch_size 16 --version_2_with_negative ; read" \; \
    new-window "cd strap-generation ; python strap/filter.py ; read" \; \
    new-window "cd strap-generation ; python strap/cleanup.py ; read" \; \
    detach \;
```
