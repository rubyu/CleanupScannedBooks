#!/usr/bin/env bash

virtualenv --no-site-packages -p /usr/bin/python2.7 virtualenv_python27
source virtualenv_python27/bin/activate

pip install pypdf2
pip install pillow
pip install chainer

mv ./patch/filters.py ./virtualenv_python27/local/lib/python2.7/site-packages/PyPDF2/

# bash ./run.sh
# watch -n 1 nvidia-smi
