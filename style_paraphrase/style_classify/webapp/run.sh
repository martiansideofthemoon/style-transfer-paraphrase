#!/bin/sh
#SBATCH --job-name=job_author_webapp
#SBATCH -o /mnt/nfs/work1/miyyer/kalpesh/projects/style-embeddings/author-classify/webapp/log.txt
#SBATCH --time=168:00:00
#SBATCH --partition=titanx-long
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --mem=1GB
#SBATCH -d singleton

cd /mnt/nfs/work1/miyyer/kalpesh/projects/style-embeddings/author-classify/webapp
export FLASK_APP=app.py
echo "Running flask app..."
python -m flask run --port 5002