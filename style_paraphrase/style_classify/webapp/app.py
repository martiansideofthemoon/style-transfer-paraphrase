"""This flask application shows the CPU usage."""
from flask import Flask, url_for, redirect
from flask import render_template
import glob
import json
import os
import pickle
import re

import math
import numpy as np
import pickle

app = Flask(__name__)

pattern = re.compile(
    r'\|\sepoch\s(\d+):\s*(\d+)\s\/\s(\d+)\sloss=(\d+\.?\d*),\snll_loss=.*,\sppl=.*,\swps=.*,\sups=.*,\swpb=.*,\sbsz=.*,\snum_updates=.*,\slr=.*,\sgnorm=.*,\sclip=.*,\soom=.*,\sloss_scale=.*,\swall=.*,\strain_wall=.*,\saccuracy=(\d+\.?\d*)'
)

pattern_valid = re.compile(
    r"\|\sepoch\s(\d+)\s\|\svalid\son\s'valid'\ssubset\s\|\sloss\s(\d+\.?\d*)\s\|\snll_loss\s.*\s\|\sppl\s.*\s\|\snum_updates\s.*\s\|\saccuracy\s(\d+\.?\d*)"
)

pattern2 = re.compile(
    r'Script Name = slurm-schedulers\/schedule_(.+)\.sh'
)

death_patterns = [
    re.compile(r'[a-zA-Z]*Error'),
    re.compile(r'ran\sout\sof\smemory'),
    re.compile(r'CANCELLED')
]


def plot_results(result_points):
    """Make a plot comparing result points."""
    # Obtaining parameter comparisons
    comparison = []
    for run_id in result_points:
        with open('../logs/expts.txt', 'r') as f:
            data = f.read().split('\n')
        lines = data[int(run_id) * 5:int(run_id) * 5 + 5]
        comparison.append(lines)

    # Code to detect errors in runs
    for i, job in enumerate(comparison):
        run_id = re.findall(pattern2, job[0])[0]
        filename = "../logs/log_%s.txt" % run_id
        with open(filename, 'r') as f:
            log_data = f.read()
        for p in death_patterns:
            death_matches = [x for x in re.findall(p, log_data)]
            if len(death_matches) > 0:
                comparison[i][0] = \
                    "<span style='color:red'><b>" + comparison[i][0] + "</b></span>"
                break

    comparison = "<br><br>".join(["<br>".join(x) for x in comparison])
    # Obtaining plotting points
    all_train_loss_data = []
    all_train_accuracy_data = []
    all_valid_accuracy_data = []
    for run_id in result_points:
        train_loss = []
        train_accuracy = []
        valid_accuracy = []
        filename = "../logs/log_%s.txt" % run_id
        with open(filename, 'r') as f:
            raw_log = f.read()

        matches = re.findall(pattern, raw_log)
        matches2 = re.findall(pattern_valid, raw_log)
        for i, match in enumerate(matches):
            train_loss.append({
                'x': int(match[0]) + float(match[1]) / float(match[2]) - 1,
                'y': float(match[3])
            })
            train_accuracy.append({
                'x': int(match[0]) + float(match[1]) / float(match[2]) - 1,
                'y': float(match[4])
            })
        for i, match in enumerate(matches2):
            valid_accuracy.append({
                'x': int(match[0]),
                'y': float(match[2])
            })
        train_loss_data = {
            "type": "spline",
            "showInLegend": True,
            "name": str(run_id),
            "dataPoints": train_loss
        }
        train_accuracy_data = {
            "type": "spline",
            "showInLegend": True,
            "name": str(run_id),
            "dataPoints": train_accuracy
        }
        valid_accuracy_data = {
            "type": "spline",
            "showInLegend": True,
            "name": str(run_id),
            "dataPoints": valid_accuracy
        }
        all_train_loss_data.append(train_loss_data)
        all_train_accuracy_data.append(train_accuracy_data)
        all_valid_accuracy_data.append(valid_accuracy_data)

    return json.dumps(all_train_loss_data), json.dumps(all_train_accuracy_data), json.dumps(all_valid_accuracy_data), comparison


@app.route('/logs/<filename>')
def logs(filename):
    """Basic usage monitoring static page."""
    filename = ''.join([x for x in filename if x.isalnum() or x == '_' or x == '.'])
    with open('../logs/%s' % filename, 'r') as f:
        data = f.read()
    data = data.replace('<', '&lt;')
    data = data.replace('>', '&gt;')
    data = data.replace('\n', '<br>')
    return data


@app.route('/schedulers/<filename>')
def schedulers(filename):
    """Basic usage monitoring static page."""
    filename = ''.join([x for x in filename if x.isalnum() or x == '_' or x == '.'])
    with open('../schedulers/%s' % filename, 'r') as f:
        data = f.read()
    data = data.replace('<', '&lt;')
    data = data.replace('>', '&gt;')
    data = data.replace('\n', '<br>')
    return data


@app.route('/results/<experiments>')
def results(experiments):
    """Basic usage monitoring static page."""
    runs = experiments.split(',')
    all_runs = []
    for run in runs:
        if '-' in run:
            limits = run.split('-')
            all_runs.extend([str(i) for i in range(int(limits[0]), int(limits[1]) + 1)])
        else:
            all_runs.append(run)
    train_loss, train_accuracy, valid_accuracy, comparison = plot_results(all_runs)
    return render_template(
        'results.html', top_left=train_accuracy, top_right=valid_accuracy, bottom_left=train_loss, bottom_right=comparison
    )
