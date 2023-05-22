#!/bin/bash

# List of loaded experiments
attackids="55 54 76 57 56 77 155 154 198 157 156 199 167 166 200 169 168 201 203 202 204 206 205 207 255 254 273 257 256 272 351 350 356 353 352 357"

# For each experiment generate a folder with the run configuration
for attackid in $attackids ;
  do
    echo "Generating runs for $attackid..."
    python generator.py "$attackid" -y -n
    echo ""
  done
