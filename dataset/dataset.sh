#!/bin/bash


curl -LJO "https://github.com/hasibzunair/3D-image-classification-tutorial/releases/download/v0.2/CT-0.zip" --output "CT-0.zip"
curl -LJO "https://github.com/hasibzunair/3D-image-classification-tutorial/releases/download/v0.2/CT-23.zip" --output "CT-23.zip"

unzip "CT-0.zip" -d "./"
unzip "CT-23.zip" -d "./"

rm "CT-0.zip"
rm "CT-23.zip"
