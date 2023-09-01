import json

json_pth = 'old_finetune_labels.json' # 'old_base_labels.json', 'old_finetune_labels.json'
data = json.load(open(json_pth, 'r'))


sorted_data = dict(sorted(data.items(), key=lambda item: item[1]))
print(sorted_data)
json_pth = json_pth.replace('old_', '')
with open('%s'%(json_pth), 'w') as outfile:
    json.dump(sorted_data, outfile)