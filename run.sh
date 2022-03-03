#!/bin/bash -e
pip install .
./linter.sh 
python -m app.main 