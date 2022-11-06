#! /bin/sh

# Zip the current folder except the assets subfolder
zip -r ../toto_starter.zip . -x assets/*
