#!/bin/bash

pip3 install dtrx

curl -LJO "https://github.com/hasibzunair/3D-image-classification-tutorial/releases/download/v0.2/CT-0.zip" --output "CT-0.zip"
curl -LJO "https://github.com/hasibzunair/3D-image-classification-tutorial/releases/download/v0.2/CT-23.zip" --output "CT-23.zip"

dtrx CT-0.zip --quiet
dtrx CT-23.zip --quiet

rm "CT-0.zip"
rm "CT-23.zip"
