import json
import os
import numpy as np
import matplotlib.pyplot as plt

rew_list = []
# Specify the path to the JSON file
# file_dir = 'C:\\Users\\jerry\\Desktop\\hly\\2024-1\\Rein\\p-0\\Check1'
file_dir = 'C:\\Users\\jerry\\Desktop\\hly\\2024-1\\Rein\\git_repos\\RL_Drone-1\\Check0'
for i in range(1,10):
    # file_name = f"rew-102045-{i}.json" #02
    # file_name = f"rew-121802-{i}.json" #22b
    file_name = f"rew-092215-{i}.json"
    file_path = os.path.join(file_dir, file_name)
    with open(file_path, 'r') as file:
        data = json.load(file)
        rew_list.extend(data)



file_dir = 'C:\\Users\\jerry\\Desktop\\hly\\2024-1\\Rein\\git_repos\\RL_Drone-1\\Check1'
for i in range(1,10):
    # file_name = f"rew-102045-{i}.json" #02
    # file_name = f"rew-121802-{i}.json" #22b
    file_name = f"rew-101525-{i}.json"
    file_path = os.path.join(file_dir, file_name)
    with open(file_path, 'r') as file:
        data = json.load(file)
        rew_list.extend(data)





file_dir = 'C:\\Users\\jerry\\Desktop\\hly\\2024-1\\Rein\\git_repos\\RL_Drone-1\\Check2b'
for i in range(0,10):
    # file_name = f"rew-102045-{i}.json" #02
    # file_name = f"rew-121802-{i}.json" #22b
    file_name = f"rew-130158-{i}.json"
    file_path = os.path.join(file_dir, file_name)
    with open(file_path, 'r') as file:
        data = json.load(file)
        rew_list.extend(data)





file_dir = 'C:\\Users\\jerry\\Desktop\\hly\\2024-1\\Rein\\git_repos\\RL_Drone-1\\Check3'
for i in range(0,7):
    # file_name = f"rew-102045-{i}.json" #02
    # file_name = f"rew-121802-{i}.json" #22b
    file_name = f"rew-131219-{i}.json"
    file_path = os.path.join(file_dir, file_name)
    with open(file_path, 'r') as file:
        data = json.load(file)
        rew_list.extend(data)

for i in range(0,3):
    # file_name = f"rew-102045-{i}.json" #02
    # file_name = f"rew-121802-{i}.json" #22b
    file_name = f"rew-131359-{i}.json"
    file_path = os.path.join(file_dir, file_name)
    with open(file_path, 'r') as file:
        data = json.load(file)
        rew_list.extend(data)

# file_name = f"rew-101704-1.json" #01
# file_path = os.path.join(file_dir, file_name)
# with open(file_path, 'r') as file:
#     data = json.load(file)
#     rew_list.extend(data)

# print(np.array(rew_list).mean())

# rew_list = []


# print(np.array(rew_list).mean())



plt.plot(rew_list)
plt.show()
# print(rew_list)
# print(np.array(rew_list).mean())