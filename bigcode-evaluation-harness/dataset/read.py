import json

d = []
with open("mbpp_python_tested.json","r") as f:
    d = json.loads(f.read())

for i,v in enumerate(d):
    pass