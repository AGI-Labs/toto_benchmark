#! /bin/sh

# Zip the current folder except the assets subfolder
zip -r ../toto_benchmark.zip . -x assets/*
