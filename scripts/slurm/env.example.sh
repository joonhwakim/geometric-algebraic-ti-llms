# Copy to .env at the top of the checkout and fill in. Sourced by every job script.
export SLURM_ACCOUNT=""                 # sbatch --account
export SLURM_PARTITION=""               # a GPU partition
export SLURM_QOS=""                     # optional
export SLURM_CONSTRAINT=""              # optional node feature (the paper's runs: one GPU type for every run)
export RUNS_ROOT=""                     # absolute results root on cluster storage; the export reads it
export HF_HOME=""                       # Hugging Face cache (model weights)
export VENV=".venv"                     # the environment with the inference extra installed (REPRODUCE.md)
