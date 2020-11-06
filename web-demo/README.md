# Reformulating Unsupervised Style Transfer as Paraphrase Generation - Website

This is the official repository for the website and demo accompanying the EMNLP 2020 long paper *[Reformulating Unsupervised Style Transfer as Paraphrase Generation](https://arxiv.org/abs/2010.05700)*. This code is heavily based off a previous project, [martiansideofthemoon/squash-website](https://github.com/martiansideofthemoon/squash-website).

## STRAP Landing Page

The landing page is a static HTML file which can be found under [`strap-landing`](strap-landing). All the code is written in [`strap-landing/index.html`](strap-landing/index.html). The website is hosted at [http://style.cs.umass.edu/](http://style.cs.umass.edu/). This file has been adapted from [Rowan Zeller](https://rowanzellers.com)'s landing page for [HellaSwag](https://rowanzellers.com/hellaswag/).

## Demo Setup

Choose an output directory to store the outputs and model checkpoints. Set it in [`config.json`](config.json) as well as [`setup.sh`](setup.sh).

## STRAP Backend

The code for the STRAP APIs and backend is found under [`strap-backend`](strap-backend). All the code is written in [`strap-backend/app.py`](strap-backend/app.py). The code requires Python 3.6+ (for the `secrets` module) as well as the python package [Flask](https://palletsprojects.com/p/flask/). This code triggers the scripts in the `demo` branch of the main strap repository. To get started,

```
cd strap-backend
export FLASK_APP=app.py
python -m flask run --host 0.0.0.0 --port 3005
```

Remove the `--host 0.0.0.0` flag if you do not want to expose the APIs publicly. Also note that you will need to restart the Flask server to reflect edits in the codebase.

Finally, in a different terminal launch the following,

```
export CUDA_VISIBLE_DEVICES=0
python demo_service.py
```

(For running all commands together, you might find the `tmux` command under **Production Level Deployment** useful)

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
    new-window "cd squash-website/squash-frontend ; npx serve -s build -l 3000 ; read" \; \
    new-window "cd strap-generation ; export CUDA_VISIBLE_DEVICES=0 ; python demo_service.py ; read" \; \
    detach \;
```
