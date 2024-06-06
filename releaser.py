import airsim

client = airsim.MultirotorClient()
client.enableApiControl(True)
client.armDisarm(True)


client.simPause(False)


