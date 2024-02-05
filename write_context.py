import json

f=open('/home/rtml/shounak_research/bevfusion/data/nuscenes/v1.0-trainval/scene.json')

data = json.load(f)
print(len(data))

for d in data:
	if "Rain" in d["description"] or "rain" in d["description"]:
		d["R"] = "1"
	else:
		d["R"] = "0"

	if "Construction" in d["description"] or "construction" in d["description"]:
		d["C"] = "1"
	else:
		d["C"] = "0"

with open("/home/rtml/shounak_research/bevfusion/data/nuscenes/v1.0-trainval/scene_mod.json", "w") as jsonFile:
    json.dump(data, jsonFile, indent=2)
