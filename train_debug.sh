#!/bin/bash
shift
python -m ipdb -c continue train.py "$@"

