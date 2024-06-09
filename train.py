import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import airsim
import os
import math
import time
import json


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
    num_episodes = 100
    max_timestep = 600
    checkpoint_path = "D:\\unreal\\git_repos\\RL_Drone\\Check"
    
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
        # self.softmax = nn.Softmax(dim=-1)
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
    def __init__(self):
        target_temp = np.random.rand(3)*140-70 # target can be inside of an object
        if target_temp[2] > -5:
            #NED coord
            target_temp[2] = 5
        self.world_pos = airsim.Vector3r(target_temp[0], target_temp[1], target_temp[2])
        acc_tmp = np.random.rand(3)*1.2 - 0.6 # +- 6.7m/ss max
        self.accel = airsim.Vector3r(acc_tmp[0], acc_tmp[1], acc_tmp[2])
        self.vel = airsim.Vector3r(0,0,0)
        self.max_speed = 10
        self.how_long = 0

    def step(self):
        is_onEdge = False
        self.how_long -= 1
        self.vel += self.accel
        # normalize
        spd = self.vel.get_length()
        if spd > self.max_speed:
            self.vel = self.vel/spd * self.max_speed

        # update pos
        self.world_pos += self.vel

        if -85 > self.world_pos.x_val:
            self.accel.x_val = 1
            is_onEdge = True
            self.how_long = 0
        elif self.world_pos.x_val > 85:
            self.accel.x_val = -1
            is_onEdge = True
            self.how_long = 0
        if -85 > self.world_pos.y_val:
            self.accel.y_val = 1
            is_onEdge = True
            self.how_long = 0
        elif self.world_pos.y_val > 85:
            self.accel.y_val = -1
            is_onEdge = True
            self.how_long = 0
        if -85 > self.world_pos.z_val:
            self.accel.z_val = 1
            is_onEdge = True
            self.how_long = 0
        elif self.world_pos.z_val > 0:
            self.world_pos.z_val = 0
            self.how_long = 0


        if self.how_long < 1 and not is_onEdge:
            self.accel = airsim.Vector3r(np.random.rand()*1.2 - 0.6, np.random.rand()*1.2 - 0.6, np.random.rand()*1.2 - 0.6)
            self.how_long = np.random.randint(5,50)


class Environment:
    #!!!!!!! check if clock speed affects move duration
    def configure(self):
        self.steps_per_second = 5
        self.max_speed = 21
        self.goal_distance = 8
        # in deg/.2s, 36 -> 180deg/s 
        self.max_yaw_rate = 36
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
        # target
        self.target = Target()
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
    
    def reset(self):
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
        responses = None


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
        



## GAE TO BE REVIEWED
def compute_gae(rewards, values, next_value, gamma, lam):
    values = values + [next_value]
    gae = 0
    returns = []
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] - values[i]
        gae = delta + gamma * lam * gae
        returns.insert(0, gae + values[i])
    return returns



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
critic = Critic(input_dim, configs.critic_hidden_size).to(device)
actor_optimizer = optim.Adam(actor.parameters(), lr=configs.lr)
critic_optimizer = optim.Adam(critic.parameters(), lr=configs.lr)

# load checkpoint data
# print("loading check point")
# check_file = "Check-291941.pth"
# checkpoint = torch.load(os.path.join("D:\\unreal\\Check", check_file))
# actor.load_state_dict(checkpoint['Actor_state_dict'])
# critic.load_state_dict(checkpoint['Critic_state_dict'])
# actor_optimizer.load_state_dict(checkpoint['optimizerA_state_dict'])
# critic_optimizer.load_state_dict(checkpoint['optimizerC_state_dict'])
# print("checking point loaded")


actor.train()
critic.train()

rewards_hist = []
hist_save_prefix = time.strftime("rew-%d%H%M-", time.localtime(time.time()))
check_file = time.strftime("Check-%d%H%M.pth", time.localtime(time.time()))

for episode in range(configs.num_episodes):
    state = env.reset()
    log_probs = []
    values = []
    rewards = []

    for _ in range(configs.max_timestep):  # Upper bound on episode length
        state_tensor = to_gpu_tensor(state)
        action_prob = actor(state_tensor).to('cpu')
        # action = action_probs.multinomial(1)
        action = [np.random.choice([0, 1], p=[1-p.item(), p.item()]) for p in action_prob.detach()]
        action = torch.Tensor(action)
        next_state, reward, done, _ = env.step(action)
        # sigmoid -> as prob foreach -> log (mul_product of (each prob))

        probab = 1 - action - action_prob
        probab = probab.abs() # chance to take this exact action
        log_probs.append(torch.log(probab).sum())
        values.append(critic(state_tensor))
        rewards.append(reward)

        if done:
            break
        else:
            state = next_state

    next_state_tensor = to_gpu_tensor(next_state)
    next_state_value = critic(next_state_tensor).item() ## item part may need review

    returns = compute_gae(rewards, [v.item() for v in values],next_state_value, configs.gamma, configs.gae_lambda)
    returns = to_gpu_tensor(returns)
    returns = returns #no detach
    values = torch.cat(values).squeeze()
    advantage = returns - values

    log_probs = torch.stack(log_probs).to(device)
    actor_loss = -torch.sum(log_probs * advantage.detach()) # watch detach
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    critic_loss = F.smooth_l1_loss(values, returns)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
    print(f"episode done: {episode}")
    rewards_hist.append(float(np.array(rewards).sum()))
    if episode % 10 == 0 and episode > 1:
        # print(f"episode: {episode}, Total Loss: {total_loss.item()}")
        print(f"episode: {episode}, Rewards: {np.array(rewards).sum()}")
        torch.save({
            'Actor_state_dict': actor.state_dict(),
            'Critic_state_dict': critic.state_dict(),
            'optimizerA_state_dict': actor_optimizer.state_dict(),
            'optimizerC_state_dict': critic_optimizer.state_dict(),
            }, os.path.join(configs.checkpoint_path, check_file))
        #save 
        hist_save_name = hist_save_prefix + str(episode//10)+".json"
        with open(os.path.join(configs.checkpoint_path, hist_save_name), 'w') as file:
            # rewards_hist_list = [int(rew) for rew in rewards_hist]
            json.dump(rewards_hist, file)
        rewards_hist = []