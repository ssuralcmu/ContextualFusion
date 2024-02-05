import json

f=open('/home/rtml/shounak_research/bevfusion/data/nuscenes/v1.0-trainval/scene.json')

data = json.load(f)
data = [x for x in data if "Night" in x["description"]]
print(len(data))

for d in data:
	if "Night" in d["description"]:
		d["W"] = "1"
	else:
		d["W"] = "0"

with open("/home/rtml/shounak_research/bevfusion/data/nuscenes/v1.0-trainval/scene.json", "w") as jsonFile:
    json.dump(data, jsonFile, indent=2)