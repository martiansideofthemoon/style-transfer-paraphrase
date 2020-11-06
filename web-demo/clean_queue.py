import json
import subprocess

with open("config.json", "r") as f:
    configuration = json.loads(f.read())
    OUTPUT_DIR = configuration["output_dir"]

command = "rm {}/generated_outputs/queue/queue.txt".format(OUTPUT_DIR)
print(subprocess.check_output(command, shell=True))

command = "touch {}/generated_outputs/queue/queue.txt".format(OUTPUT_DIR)
print(subprocess.check_output(command, shell=True))
