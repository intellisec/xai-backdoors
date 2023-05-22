#!/bin/bash

#export CUBLAS_WORKSPACE_CONFIG=:4096:8
while :
do
  python worker.py "$@"
  sleep 10
done