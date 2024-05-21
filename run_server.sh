#!/bin/bash

zenml down
zenml up

export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

python run_deployment.py --config deploy
python run_deployment.py --config predict

streamlit run streamlit_app.py 
