#!/bin/bash

rm -rf deployment/
mkdir deployment/

# Copy relevant code.
cp -r app deployment/
cp -r aws deployment/

# Install dependencies.
pip install --no-deps -r requirements.txt -t deployment/

# Zip contents.
zip -r lambda_package.zip deployment/
