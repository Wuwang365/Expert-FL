import subprocess
import os
import time
import json

names = os.listdir("data/traindata/")
with open('data/info.json','r') as f:
    names = json.loads(f.read()).keys()

for name in names:
    cmd = f"python client_wapper.py --name {name}"
    cmd = cmd.split(' ')
    subprocess.Popen(cmd, shell=False)
    time.sleep(0.1)