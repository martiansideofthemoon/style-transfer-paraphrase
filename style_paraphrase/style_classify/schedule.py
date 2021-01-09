"""Schedule jobs for author classification."""
import itertools
import collections
import glob
import os
import datetime
import subprocess
import string
import sys


def get_run_id():
    """Look up global database of jobs."""
    filename = "style_paraphrase/style_classify/logs/expts.txt"
    if os.path.isfile(filename) is False:
        with open(filename, 'w') as f:
            f.write("")
        return 0
    else:
        with open(filename, 'r') as f:
            expts = f.readlines()
        run_id = len(expts) / 5
    return run_id

top_details = "Grid search over style classification with different num_classes"

dataset_dependencies_default = {
    "size": 166252,
    "num_classes": 37
}

dataset_dependencies = {
    "shakespeare": {
        "size": 36790,
        "num_classes": 2
    },
    "formality": {
        "size": 105169,
        "num_classes": 2
    },
    "cola": {
        "size": 8551,
        "num_classes": 2
    },
    "cds": {
        "size": 273372,
        "num_classes": 11
    },
}

other_dependencies = {
    "total_updates": lambda x: int((x["size"] * x["num_epochs"]) // (x["max_sentences"] * x["update_freq"])),
    "warmup": lambda x: int(0.06 * ((x["size"] * x["num_epochs"]) // (x["max_sentences"] * x["update_freq"]))),
}

hyperparameters = [
    [('base_dataset',), ["datasets"]],
    [('dataset',), ["cds"]],
    [('roberta_model',), ['LARGE']],
    [('learning_rate',), ['1e-5']],
    [('num_epochs',), [10]],
    [('max_sentences',), [4]],
    [('update_freq',), [8]],
    [('max_positions',), [512]],
]


run_id = int(get_run_id())
key_hyperparameters = [x[0] for x in hyperparameters]
value_hyperparameters = [x[1] for x in hyperparameters]
combinations = list(itertools.product(*value_hyperparameters))

scripts = []

for combo in combinations:
    # Write the scheduler scripts
    with open("style_paraphrase/style_classify/author_classify_template.sh", 'r') as f:
        schedule_script = f.read()
    combo = {k[0]: v for (k, v) in zip(key_hyperparameters, combo)}
    # populate dataset dependenices
    if combo['dataset'] not in dataset_dependencies:
        ddepend = dataset_dependencies_default
    else:
        ddepend = dataset_dependencies[combo['dataset']]

    for k, v in ddepend.items():
        combo[k] = v
    for k, v in other_dependencies.items():
        combo[k] = v(combo)

    od = collections.OrderedDict(sorted(combo.items()))
    lower_details = ""
    for k, v in od.items():
        lower_details += "%s = %s, " % (k, str(v))
    # removing last comma and space
    lower_details = lower_details[:-2]

    combo["top_details"] = top_details
    combo["lower_details"] = lower_details
    combo["job_id"] = run_id
    print("Scheduling Job #%d" % run_id)

    for k, v in combo.items():
        if "{%s}" % k in schedule_script:
            schedule_script = schedule_script.replace("{%s}" % k, str(v))

    schedule_script += "\n"
    # Write schedule script
    script_name = 'style_paraphrase/style_classify/slurm-schedulers/schedule_%d.sh' % run_id
    with open(script_name, 'w') as f:
        f.write(schedule_script)
    scripts.append(script_name)

    # Making files executable
    subprocess.check_output('chmod +x %s' % script_name, shell=True)

    # Update experiment logs
    output = "Script Name = " + script_name + "\n" + \
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + "\n" + \
        top_details + "\n" + \
        lower_details + "\n\n"
    with open("style_paraphrase/style_classify/logs/expts.txt", "a") as f:
        f.write(output)
    # For the next job
    run_id += 1

# schedule jobs
for script in scripts:
    command = "sbatch %s" % script
    print(subprocess.check_output(command, shell=True))
