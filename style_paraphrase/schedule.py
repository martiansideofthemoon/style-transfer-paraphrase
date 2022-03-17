import collections
import datetime
import itertools
import os
import subprocess

from hyperparameters_config import (paraphrase, inverse_paraphrase)


class SafeDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'


def get_run_id():
    filename = "style_paraphrase/logs/expts.txt"
    if os.path.isfile(filename) is False:
        with open(filename, 'w') as f:
            f.write("")
        return 0
    else:
        with open(filename, 'r') as f:
            expts = f.readlines()
        run_id = len(expts) / 5
    return run_id


other_dependencies = {
    "memory": lambda x: int(x["ngpus"]) * 50 if x["gpu"] in ["m40", "titanx"] else int(x["ngpus"]) * 45,
    "cpus": lambda x: int(x["ngpus"]) * 3
}


top_details = "GPT2 model for formality."
hyperparameters = inverse_paraphrase

run_id = int(get_run_id())
key_hyperparameters = [x[0] for x in hyperparameters]
value_hyperparameters = [x[1] for x in hyperparameters]
combinations = list(itertools.product(*value_hyperparameters))

scripts = []
eval_scripts = []

for combo in combinations:
    # Write the scheduler scripts
    with open("style_paraphrase/run_finetune_gpt2_template.sh", 'r') as f:
        schedule_script = f.read()
    with open("style_paraphrase/run_evaluate_gpt2_template.sh", 'r') as f:
        evaluate_script = f.read()

    combo = {k[0]: v for (k, v) in zip(key_hyperparameters, combo)}

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

    for k, v in combo.items():
        if "{%s}" % k in evaluate_script:
            evaluate_script = evaluate_script.replace("{%s}" % k, str(v))

    schedule_script += "\n"
    evaluate_script += "\n"

    # Write schedule script
    script_name = 'style_paraphrase/slurm-schedulers/schedule_%d.sh' % run_id
    with open(script_name, 'w') as f:
        f.write(schedule_script)

    evaluate_script_name = 'style_paraphrase/slurm-schedulers/evaluate_%d.sh' % run_id
    with open(evaluate_script_name, 'w') as f:
        f.write(evaluate_script)

    scripts.append(script_name)
    eval_scripts.append(evaluate_script_name)

    # Making files executable
    subprocess.check_output('chmod +x %s' % script_name, shell=True)
    subprocess.check_output('chmod +x %s' % evaluate_script_name, shell=True)

    # Update experiment logs
    output = "Script Name = " + script_name + "\n" + \
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + "\n" + \
        top_details + "\n" + \
        lower_details + "\n\n"
    with open("style_paraphrase/logs/expts.txt", "a") as f:
        f.write(output)
    # For the next job
    run_id += 1


# schedule jobs
for script in scripts:
    command = "sbatch %s" % script
    print(subprocess.check_output(command, shell=True))

for script in eval_scripts:
    command = "sbatch %s" % script
    print(subprocess.check_output(command, shell=True))
