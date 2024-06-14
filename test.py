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
    num_episodes = 4
    #!!!!!
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


class Target:
    def __init__(self,i):
        # target_temp = np.random.rand(3)*140-70 # target can be inside of an object
        # if target_temp[2] > -5:
        #     #NED coord
        #     target_temp[2] = 5
        # self.world_pos = airsim.Vector3r(target_temp[0], target_temp[1], target_temp[2])
        signX = i%2*2-1                 #-1 1 -1 1
        signY = math.floor(i/2)*2-1     #-1 -1 1 1
        self.world_pos = airsim.Vector3r(80 * signX, 80*signY, -10)
        # acc_tmp = np.random.rand(3)*1.2 - 0.6 # +- 6.7m/ss max
        acc_tmp = np.random.rand(3)*1.2 - 0.6 # +- ?m/ss max
        #!!!!!!!!
        self.accel = airsim.Vector3r(acc_tmp[0], acc_tmp[1], acc_tmp[2])
        self.vel = airsim.Vector3r(0,0,0)
        self.max_speed = 0
        #!!!!!!!!
        self.how_long = 0

    def step(self):
        is_onEdge = False
        self.how_long -= 1
        self.vel += self.accel
        # normalize
        spd = self.vel.get_length()
        if spd > self.max_speed:
            self.vel = self.vel/spd * self.max_speed

        # update pos !!!!!
        # self.world_pos += self.vel

        if -80 > self.world_pos.x_val:
            self.accel.x_val = 1.2
            is_onEdge = True
            self.how_long = 0
        elif self.world_pos.x_val > 80:
            self.accel.x_val = -1.2
            is_onEdge = True
            self.how_long = 0
        if -80 > self.world_pos.y_val:
            self.accel.y_val = 1.2
            is_onEdge = True
            self.how_long = 0
        elif self.world_pos.y_val > 80:
            self.accel.y_val = -1.2
            is_onEdge = True
            self.how_long = 0
        if -70 > self.world_pos.z_val:
            self.accel.z_val = 1.2
            is_onEdge = True
            self.how_long = 0
        elif self.world_pos.z_val > 0:
            self.world_pos.z_val = 0
            self.how_long = 0


        if self.how_long < 1 and not is_onEdge:
            self.accel = airsim.Vector3r(np.random.rand()*1.2 - 0.6, np.random.rand()*1.2 - 0.6, np.random.rand()*1.2 - 0.6)
            # !!!!!
            self.how_long = np.random.randint(10,50)


class Environment:
    def configure(self):
        self.steps_per_second = 5
        self.max_speed = 21
        self.goal_distance = 8
        # !!!!!!!!^8
        # in deg/.2s, 36 -> 180deg/s 
        self.max_yaw_rate = 36
        self.tolerance = 2
        #!!!!!!!^2
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
        # target
        self.target = Target(0)
        # wind
        self.wind_vec = airsim.Vector3r(np.random.rand()*10 - 5, np.random.rand()*10 - 5, np.random.rand()*10 - 5)
        self.half_windd = self.wind_d_rate/2
        wind_amp = self.wind_vec.get_length()
        if wind_amp > self.max_wind:
            self.wind_vec / wind_amp * self.max_wind
        # time and enter data
        self.t_in = -1
        self.t_step = 0
        self.is_in = False
        # dimensions
        self.action_dim = 3 + 3 + 3 + 3 #vx vy vz yaw
        self.state_dim = 3+30+3+3+1 #target(in local) sensor wind me(veloxyz, avel-yaw)
        # setup client cam
        camera_pose_l = airsim.Pose(airsim.Vector3r(0, 0, -.5), airsim.to_quaternion(0, 0, math.radians(-90))) #radians
        camera_pose_r = airsim.Pose(airsim.Vector3r(0, 0, -.5), airsim.to_quaternion(0, 0, math.radians(90))) #radians
        self.client.simSetCameraPose("front_left", camera_pose_l)
        self.client.simSetCameraPose("front_right", camera_pose_r)
    
    def reset(self,i):
        self.target = Target(i)
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
        return self.get_state((0,0,-1,0))

    def get_state(self, last_vel):
        state = [] # target x, y ,z / closest front, left, right, back, bottom, top / wind x y z / prevmove vxyz yaw

        # target coord in local
        local_target = self.target.world_pos - self.client.simGetVehiclePose().position
        world_ori = self.client.simGetVehiclePose().orientation.inverse()
        world_ori_conj = world_ori.conjugate()
        local_target_q = local_target.to_Quaternionr()
        local_target = world_ori * local_target_q * world_ori_conj
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

        # images = np.array([])
        for response in responses:
            img = np.array(response.image_data_float).reshape((40,40))
            #img size 40 x 40, divide by 9 22 9
            state.append(np.min(img[:, :9]))
            state.append(np.min(img[:9, 9:31]))
            state.append(np.min(img[9:31, 9:31]))
            state.append(np.min(img[31:, 9:31]))
            state.append(np.min(img[:, 31:]))


        # wind in local
        local_wind_q = self.wind_vec.to_Quaternionr()
        local_wind = world_ori * local_wind_q * world_ori_conj
        state.append(local_wind.x_val)
        state.append(local_wind.y_val)
        state.append(local_wind.z_val)

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
        # update target
        self.target.step()
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
            inner = self.goal_distance - self.tolerance
            outter = self.goal_distance + self.tolerance
            local_target_norot = self.target.world_pos - self.client.simGetVehiclePose().position
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
                    print(f"entered! step: {self.t_in}")
                    self.was_in = True
                reward = np.log(self.t_step - self.t_in + np.e)+1.4
                #!!!!!!
            else:
                # too close
                print(f"too close, step: {self.t_step}")
                self.was_in = False
                reward = -1/(target_deviation/inner + 0.5)
        
        my_pos = self.client.simGetVehiclePose().position
        if my_pos.x_val < -100 or 100 < my_pos.x_val or my_pos.y_val < -100 or 100 < my_pos.y_val or my_pos.z_val < -85:
            # out of env, too high up
            print(f"out of env step: {self.t_step}")
            done = True
        return self.get_state((vx, vy, vz, vyaw)), reward, done, None
        


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
check_file = "Check-131219.pth"
checkpoint = torch.load(os.path.join("D:\\unreal\\git_repos\\RL_Drone\\Check", check_file))

# checkpoint = torch.load(os.path.join(configs.checkpoint_path, check_file))
actor.load_state_dict(checkpoint['Actor_state_dict'])
# critic.load_state_dict(checkpoint['Critic_state_dict'])
# actor_optimizer.load_state_dict(checkpoint['optimizerA_state_dict'])
# critic_optimizer.load_state_dict(checkpoint['optimizerC_state_dict'])
print("checking point loaded")


actor.eval()

rewards_hist = []

for episode in range(configs.num_episodes):
    state = env.reset(episode)
    rewards = []

    for _ in range(configs.max_timestep):  # Upper bound on episode length
        state_tensor = to_gpu_tensor(state)
        with torch.no_grad():
          action_prob = actor(state_tensor).to('cpu')
        probs = F.hardtanh(action_prob, 0.15, 0.85)
        probs = (probs - 0.15) * 1.4285 # mapping range into  [0, 1]

        action = [np.random.choice([0, 1], p=[1-p.item(), p.item()]) for p in probs.detach()]
        # action = [0,1,1, 0,1,1, 0,1,1, 0,0,0]
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