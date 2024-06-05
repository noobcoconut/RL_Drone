import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import airsim
import os
import math
import time
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using {device}")

## Helper function to convert to tensor
def to_tensor(ndarray, dtype=torch.float):
    return torch.tensor(ndarray, dtype=dtype)

to_gpu_tensor = lambda x:to_tensor(x).to(device)


class Configurations:
    lr = 0.001
    gamma = 0.99
    gae_lambda = 0.95
    actor_hidden_size = 256
    critic_hidden_size = 256
    num_episodes = 3
    max_timestep = 600
    checkpoint_path = "D:\\unreal\\Check"
    
    def __init__(self):
        pass

configs = Configurations()

# Actor Critic
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size, bias=False)
        self.bn1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bn2 = nn.LayerNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.bn1(F.leaky_relu(self.fc1(x), 0.02))
        x = self.bn2(F.relu(self.fc2(x)))
        return self.sigmoid(self.fc3(x))

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size, bias=False)
        self.bn1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bn2 = nn.LayerNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.bn1(F.leaky_relu(self.fc1(x), 0.02))
        x = self.bn2(F.relu(self.fc2(x)))
        return self.fc3(x)

class Environment:
    #!!!!!!! check if clock speed affects move duration
    def configure(self):
        self.steps_per_second = 5
        self.max_speed = 21
        # in deg/.2s, 36 -> 180deg/s 
        self.max_yaw_rate = 36
        self.target_distance = 8
        self.tolerance = 2 
        self.max_wind = 5
        self.wind_d_rate = 0.1

    def __init__(self):
        # airsim
        self.client = airsim.MultirotorClient()
        # client.confirmConnection()
        self.configure()
        self.step_duration = 1/self.steps_per_second
        self.action_mul = torch.tensor([4,2,1,]*4)
        self.speed_step = self.max_speed/3
        self.yaw_step = self.max_yaw_rate/3
        self.wind_vec = airsim.Vector3r(np.random.rand()*10 - 5, np.random.rand()*10 - 5, np.random.rand()*10 - 5)
        self.half_windd = self.wind_d_rate/2
        wind_amp = self.wind_vec.get_length()
        if wind_amp > self.max_wind:
            self.wind_vec / wind_amp * self.max_wind
        self.t_in = -1
        self.t_step = 0
        self.is_in = False
        self.action_dim = 3 + 3 + 3 + 3 #vx vy vz yaw
        self.state_dim = 3+6+3+3+1 #target(in local) sensor wind me(veloxyz, avel-yaw)
        # setup client cam
        camera_pose_l = airsim.Pose(airsim.Vector3r(0, 0, -.5), airsim.to_quaternion(0, 0, math.radians(-90))) #radians
        camera_pose_r = airsim.Pose(airsim.Vector3r(0, 0, -.5), airsim.to_quaternion(0, 0, math.radians(90))) #radians
        self.client.simSetCameraPose("front_left", camera_pose_l)
        self.client.simSetCameraPose("front_right", camera_pose_r)
    
    def reset(self):
        target_temp = np.random.rand(3)*160-80 # target can be inside of an object
        if target_temp[2] > 0:
            #NED coord
            target_temp[2] = 0
        # self.target_world = airsim.Vector3r(target_temp[0], target_temp[1], target_temp[2])
        self.target_world = airsim.Vector3r(50,50,-10)
        # print(f"target:{self.target_world}")
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.simPause(False)
        time.sleep(0.3) # wait for it to hit the reset pedestal
        self.client.moveToPositionAsync(0,0,-2,2, 3).join()
        self.client.simGetCollisionInfo() # for some reason, first call says it collided with the reset pedestal
        self.wind_vec = airsim.Vector3r(np.random.rand()*10 - 5, np.random.rand()*10 - 5, np.random.rand()*10 - 5)
        wind_amp = self.wind_vec.get_length()
        if wind_amp > self.max_wind:
            self.wind_vec / wind_amp * self.max_wind
        self.t_in = -1
        self.t_step = 0
        self.was_in = False
        return self.get_state((0,0,-2,0))
        # return self.get_state((0,0,0,0))

    def get_state(self, last_vel):
        state = [] # target x, y ,z / closest front, left, right, back, bottom, top / wind x y z / prevmove vxyz yaw

        # target coord in local
        local_target = self.target_world - self.client.simGetVehiclePose().position
        world_ori = self.client.simGetVehiclePose().orientation.inverse()
        local_target_q = local_target.to_Quaternionr()
        local_target = world_ori * local_target_q * world_ori.conjugate()
        state.append(local_target.x_val)
        state.append(local_target.y_val)
        state.append(local_target.z_val)

        # 6 axis distance
        responses = self.client.simGetImages([
        airsim.ImageRequest("front_center", airsim.ImageType.DepthPerspective, True, False),
        airsim.ImageRequest("front_left", airsim.ImageType.DepthPerspective, True, False),
        airsim.ImageRequest("front_right", airsim.ImageType.DepthPerspective, True, False),
        airsim.ImageRequest("back_center", airsim.ImageType.DepthPerspective, True, False),
        airsim.ImageRequest("bottom_center", airsim.ImageType.DepthPerspective, True, False),
        airsim.ImageRequest("top_center", airsim.ImageType.DepthPerspective, True, False),])

        for response in responses:
            state.append( np.min(response.image_data_float) )

        # wind
        state.append(self.wind_vec.x_val)
        state.append(self.wind_vec.y_val)
        state.append(self.wind_vec.z_val)

        # last actions in v xyz yaw
        state.extend(last_vel)
        return np.array(state)

    def step(self, action):
        # action interpretation
        self.t_step += 1
        done = False
        m_action = action*self.action_mul
        act_tensors = torch.split(m_action, 3)
        vel_step = []
        for act_t in act_tensors:
            vel_step.append(act_t.sum().item())
        vx = min(vel_step[0] * self.speed_step - self.max_speed, self.max_speed)
        vy = min(vel_step[1] * self.speed_step - self.max_speed, self.max_speed)
        vz = min(vel_step[2] * self.speed_step - self.max_speed, self.max_speed)
        vyaw = min(vel_step[3] * self.yaw_step - self.max_yaw_rate, self.max_yaw_rate)
        speed = math.sqrt(vx*vx + vy * vy + vz*vz)
        if speed > self.max_speed:
            factor = self.max_speed/speed
            vx *= factor
            vy *= factor
            vz *= factor
        # exert the action
        self.client.simContinueForTime(self.step_duration)
        self.client.moveByVelocityBodyFrameAsync(vx, vy, vz, duration = self.step_duration + 0.1, yaw_mode= airsim.YawMode(is_rate = True, yaw_or_rate = vyaw))
        time.sleep(self.step_duration/3)
        # update wind
        d_vec = airsim.Vector3r(np.random.rand()*self.wind_d_rate - self.half_windd, np.random.rand()*self.wind_d_rate - self.half_windd, np.random.rand()*self.wind_d_rate - self.half_windd)
        self.wind_vec += d_vec
        wind_amp = self.wind_vec.get_length()
        if wind_amp > self.max_wind:
            self.wind_vec / wind_amp * self.max_wind
        self.client.simSetWind(self.wind_vec)

        # calc reward
        # collision check
        col_info = self.client.simGetCollisionInfo()
        if(col_info.has_collided and self.t_step > 1):
            # and col_info.object_id != 164 exclusion for reset pedestal
            reward = -10
            print(f"collided! step:{self.t_step}")
            done = True
        else:
            inner = self.target_distance - self.tolerance
            outter = self.target_distance + self.tolerance
            #TODO optimize it
            local_target_norot = self.target_world - self.client.simGetVehiclePose().position
            target_deviation = local_target_norot.get_length()
            if target_deviation > outter:
                # a bit too far
                if self.was_in:
                    print(f"exited step: {self.t_step}")
                self.was_in = False
                reward = 1/(target_deviation - outter + 1)
            elif inner < target_deviation:
                # just right
                if(not self.was_in):
                    # update last in time
                    self.t_in = self.t_step
                    print(f"enterted! step: {self.t_in}")
                    self.was_in = True
                reward = np.log(self.t_step - self.t_in + np.e)+1
            else:
                # too close
                print(f"too close, step: {self.t_step}")
                self.was_in = False
                reward = -1/(target_deviation/inner + 0.5)
        
        my_pos = self.client.simGetVehiclePose().position
        if my_pos.x_val < -100 or 100 < my_pos.x_val or my_pos.y_val < -100 or 100 < my_pos.y_val or my_pos.z_val < -100:
            # out of env, too high up
            done = True
        return self.get_state((vx, vy, vz, vyaw)), reward, done, None
        # return self.get_state((0,0,0,0)), reward, done, None


class DummyEnv:
    def __init__(self):
        # self.observation_space = np.array([0,1,2,3])
        self.state_dim = 4
        self.action_dim = 8
    def reset(self):
        return np.array([0,1,2,3])
    def step(self,action):
        return (np.array([0,1,2,3]), 1, False, None)


# training TO BE REVIEWED
env = Environment()
# env = DummyEnv()

input_dim = env.state_dim
output_dim = env.action_dim
actor = Actor(input_dim, output_dim, configs.actor_hidden_size).to(device)

# load checkpoint data
print("loading check point")
check_file = "Check-291843.pth"
checkpoint = torch.load(os.path.join(configs.checkpoint_path, check_file))
actor.load_state_dict(checkpoint['Actor_state_dict'])
# critic.load_state_dict(checkpoint['Critic_state_dict'])
# actor_optimizer.load_state_dict(checkpoint['optimizerA_state_dict'])
# critic_optimizer.load_state_dict(checkpoint['optimizerC_state_dict'])
print("checking point loaded")


actor.eval()

rewards_hist = []

for episode in range(configs.num_episodes):
    state = env.reset()
    rewards = []

    for _ in range(configs.max_timestep):  # Upper bound on episode length
        state_tensor = to_gpu_tensor(state)
        with torch.no_grad():
          action_prob = actor(state_tensor).to('cpu')
        action = [np.random.choice([0, 1], p=[1-p.item(), p.item()]) for p in action_prob.detach()]
        # action = [round(p.item()) for p in action_prob.detach()]
        action = torch.Tensor(action)
        next_state, reward, done, _ = env.step(action)
        # sigmoid -> as prob foreach -> log (mul_product of (each prob))
        rewards.append(reward)

        if done:
            break
        else:
            state = next_state

    rewards_hist.append(float(np.array(rewards).sum()))
    print(f"episode done: {episode}, Rewards: {np.array(rewards).sum()}")

print(rewards_hist)