import subprocess

# invoke a shell command that zips the current folder except the assets subfolder
cmd = "zip -r ../toto_starter.zip . -x assets/*"
# execute the command in subprocess
subprocess.call(cmd, shell=True)