#!/bin/bash

# Replace with your Conda environment name
ENV_NAME="cscg"

# Activate the Conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Export the environment to a YAML file
conda env export > environment.yml

# Convert the YAML file to requirements.txt
python - <<EOF
import yaml

with open("environment.yml", "r") as stream:
    try:
        env = yaml.safe_load(stream)
        dependencies = env.get('dependencies', [])
        pip_dependencies = []

        for dep in dependencies:
            if isinstance(dep, dict) and dep.get('pip'):
                pip_dependencies.extend(dep['pip'])
            elif isinstance(dep, str):
                pip_dependencies.append(dep)

        with open("requirements.txt", "w") as req_file:
            for dep in pip_dependencies:
                req_file.write(dep + '\n')

    except yaml.YAMLError as exc:
        print(exc)
EOF

# Clean up intermediate files if needed
rm environment.yml

echo "requirements.txt has been created successfully."

