import os
import time
import math
import random
import numpy as np
import pybullet as p
import pybullet_data
import env_utils as utils
import matplotlib.pyplot as plt
from collections import deque
from gym.spaces import Box

ACTION_RANGE = 1.0
ACTION_DIST = 0.03
ACC_SCALE = 0.6        # m/s^2

MIN_DEP = 0.6
MAX_DEP = 1.0
SEG_NUM = 6
img_size = 256

input_size = 240


class Panda():

	def __init__(self, args):

		if args.GUI =='GUI':
			self.serverMode = p.GUI # GUI/DIRECT
		else:
			self.serverMode = p.DIRECT

		# Environment setting
		self.loadURDF_all()
		self.loadObject()
		self.camera_setting()
		self.index()
		self.canonical_color()
		
		# Data shape
		self.task_num = args.num_task
		obs_shape = (
			3, img_size, img_size)
		self.observation_space = Box(
			0, 255, shape=obs_shape, dtype=np.uint8)
		seg_shape = (
			6, img_size, img_size)
		self.seg_space = Box(
			0, 1, shape=seg_shape, dtype=np.uint8)
		dep_shape = (
			1, img_size, img_size)
		self.dep_space = Box(
			0, 1, shape=dep_shape, dtype=np.float32)

		self.action_space = Box(-ACTION_RANGE, ACTION_RANGE, (3,))
		self.hybrid_space = Box(-2, 2, (self.task_num + 8,))
		self.gt_state_space = Box(-2, 2, (29,))
		self._max_episode_steps = 100

		# Task setting
		self.laptop_threshold = 0.07
		self.drawer_threshold = 0.023
		self.threshold = [self.laptop_threshold, self.drawer_threshold]
		
		self.laptop_joint = (0,)
		self.laptop_angle = 0
		self.drawer_joint1 = 0
		self.drawer_joint2 = 0
		self.drawer_joint = (2,0,)		

		self.action_point_link = [1, 4]
		self.penalty_point_link = [2, 3]
		
		self.epi = 0
		self.eefID = 11  # ee_link
		self.init_ee_ori = p.getQuaternionFromEuler([3.141592, 0.0, 3.141592])
		self.init_jointPositions=[0.0, 0.1, 0.0, -1.5, 0.0, 1.6,  0.7963762844]
		self.ee_pos = np.array([0.0, 0.0, 0.0])

	def loadURDF_all(self):

		# connect to engine servers
		self.physicsClient = p.connect(self.serverMode)

		# add search path for loadURDFs
		p.setAdditionalSearchPath(pybullet_data.getDataPath())
		p.setGravity(0, 0, -9.8)

		# Load wall
		wall1Orientation = p.getQuaternionFromEuler([0, 0, 1.57079632679])
		wall2Orientation = p.getQuaternionFromEuler([0, 0, 1.57079632679])
		self.wall1Id = p.loadURDF("./urdf/objects/wall/wall.urdf", [ 0.584 + 0.3 + 0.025 , 0.0, 1.04], useFixedBase = True)
		self.wall2Id = p.loadURDF("./urdf/objects/wall/wall.urdf", [ 0.3+ 0.584, -0.625, 1.04], wall1Orientation, useFixedBase = True)
		self.wall3Id = p.loadURDF("./urdf/objects/wall/wall.urdf", [ 0.3+ 0.584,  0.625, 1.04], wall2Orientation, useFixedBase = True)

		# Load plane
		self.planeID = p.loadURDF("plane.urdf")

		# Load camera
		cameraStartPos = [0.0, 0.40, 1.0]
		cameraStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
		self.cameraID = p.loadURDF("./urdf/objects/camera/camera.urdf", cameraStartPos, cameraStartOrientation,useFixedBase = True, flags=p.URDF_USE_INERTIA_FROM_FILE)

		# Load table
		tableStartPos = [0.584, 0.0, 0.524]
		tableStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
		self.tableID = p.loadURDF("./urdf/objects/table/table.urdf", tableStartPos, tableStartOrientation,useFixedBase = True, flags=p.URDF_USE_INERTIA_FROM_FILE)

		# Load laptop
		self.laptopStartPos = [0.68, -0.1, 0.57]
		self.pos = self.laptopStartPos
		self.objectStartOrientation = p.getQuaternionFromEuler([0 ,0, 1.57])
		self.laptopId = p.loadURDF("./urdf/objects/laptop/laptop.urdf", self.laptopStartPos, self.objectStartOrientation, useFixedBase = True, flags=p.URDF_USE_SELF_COLLISION)
		p.setJointMotorControl2(self.laptopId, 0, p.VELOCITY_CONTROL, force=1)	

		# Load drawer
		self.drawerStartPos = [0.68, 0.2, 0.57]
		self.drawerStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
		self.drawerId = p.loadURDF("./urdf/objects/drawer/drawer.urdf", self.drawerStartPos, self.drawerStartOrientation,globalScaling=0.94,useFixedBase = True,flags=p.URDF_USE_SELF_COLLISION)
		p.setJointMotorControl2(self.drawerId, 0, p.VELOCITY_CONTROL, force=5)
		p.setJointMotorControl2(self.drawerId, 2, p.VELOCITY_CONTROL, force=5)

		self.objectId = [self.laptopId, self.drawerId]

		# setup panda with robotiq 85
		robotStartPos = [0.0, 0.0, 0.554]
		robotStartOrn = p.getQuaternionFromEuler([0, 0, 0])

		self.PandaUrdfPath = "./urdf/franka_panda/panda.urdf"
		print("----------------------------------------")
		print("Loading robot from {}".format(self.PandaUrdfPath))
		self.robotID = p.loadURDF(self.PandaUrdfPath, robotStartPos, robotStartOrn,useFixedBase = True)
							 # flags=p.URDF_USE_INERTIA_FROM_FILE)
		self.joints, self.controlJoints = utils.setup_panda(p, self.robotID)

	def loadObject(self):
		
		# Load lotion
		self.lotionStartPos = [0.48, -0.1, 0.57]
		self.objectStartOrientation = p.getQuaternionFromEuler([0 ,0, 0])
		self.lotionId = p.loadURDF("./urdf/objects/lotion/lotion.urdf", self.lotionStartPos, self.objectStartOrientation, flags=p.URDF_USE_SELF_COLLISION, useFixedBase = True)

		# Load bottle_body
		self.bottleStartPos = [0.58, -0.1, 0.57]
		self.objectStartOrientation = p.getQuaternionFromEuler([0 ,0, 0])
		self.bottleId = p.loadURDF("./urdf/objects/tumbler/tumbler.urdf", self.bottleStartPos, self.objectStartOrientation, flags=p.URDF_USE_SELF_COLLISION, useFixedBase = True)

		# Load bottle_cap
		self.bottle_capStartPos = [0.48, -0.3, 0.57]
		self.objectStartOrientation = p.getQuaternionFromEuler([0 ,0, 0])
		self.bottle_capId = p.loadURDF("./urdf/objects/tumbler_cover/tumbler_cover.urdf", self.bottle_capStartPos, self.objectStartOrientation, flags=p.URDF_USE_SELF_COLLISION, useFixedBase = True)

		
		# Load bin_body
		self.binStartPos = [0.78, -0.1, 0.57]
		self.objectStartOrientation = p.getQuaternionFromEuler([0 ,0, 0])
		self.binId = p.loadURDF("./urdf/objects/bin/bin.urdf", self.binStartPos, self.objectStartOrientation, flags=p.URDF_USE_SELF_COLLISION, useFixedBase = True)

		# Load bin_cover
		self.bin_coverStartPos = [0.6, 0.3, 0.555]
		self.objectStartOrientation = p.getQuaternionFromEuler([0 ,0, 0])
		self.bin_coverId = p.loadURDF("./urdf/objects/bin_cover/bin_cover.urdf", self.bin_coverStartPos, self.objectStartOrientation, flags=p.URDF_USE_SELF_COLLISION, useFixedBase = True)

		self.obj_id_list = [self.binId, self.bin_coverId, self.bottleId, self.bottle_capId, self.lotionId]
		

	############################ Init process ############################


	def reset(self):

		self.epi +=1
		self.trial = 0
		self.steps = 0

		# self.stage1=False

		self.task_state = np.zeros((self.task_num))
		self.task_id = random.randint(0, 1)
		self.task_id = 0
		self.task_state[self.task_id] = 1
		if self.task_id == 0:
			self.init_jointPositions = [0.0, 0.4, 0.0, -1.2, 0.0, 1.6, 0.7963762844]
		else:
			self.init_jointPositions = [0.0, -0.4, 0.0, -2.0, 0.0, 1.6, 0.7963762844]
		eps = random.uniform(0,1)
		
		while True:			
			self.laptop_pos = np.array([random.uniform(0.56, 0.67), random.uniform(-0.26,0.30), 0.55])
			self.drawer_pos = np.array([random.uniform(0.58,0.70),random.uniform(-0.35,0.35), 0.545])
			if self.task_id == 0 and eps < 0.5:
				self.drawer_pos[2] = 0
			elif self.task_id == 1 and eps < 0.5:
				self.laptop_pos[2] = 0
			if(self.cal_distance(np.array(self.laptop_pos), np.array(self.drawer_pos)) > 0.38):
				break
	
		self.laptop_ori = random.uniform(0.5, 3.141592-0.5)		
		self.laptopStartOrientation = p.getQuaternionFromEuler([0.0 ,0, self.laptop_ori])
		
		if self.drawer_pos[1] < 0:
			self.drawer_ori = random.uniform(3.141592/2.0  , 3.141592 +3.141592/2.0*(1+self.drawer_pos[1]/0.35))
		else: 
			self.drawer_ori = random.uniform(-3.141592 -3.141592/2.0*(1-self.drawer_pos[1]/0.35), -3.141592/2.0)

		self.laptop_angle = random.uniform( 0.9, 1.67)
		self.drawer_joint1 = random.uniform(0.07, 0.11)
		self.drawer_joint2 = 0

		eps = random.uniform(0,1)
		if self.task_id == 0 and eps < 0.08:
			self.drawer_joint1 = 0
		elif self.task_id == 1 and eps < 0.08:
			self.laptop_angle = 0
		
		self.drawerStartOrientation = p.getQuaternionFromEuler([0.0 ,0, self.drawer_ori])
		
		self.object_ori = [self.laptop_ori, self.drawer_ori]
		
		# self.reload()
		self.home_pose()
		self.selectObject()
		self.set_color()


		self.state, task_ee_state, seg_img, dep_img, gt_state = self.get_state()

		return self.state, task_ee_state, seg_img, dep_img, gt_state


	def home_pose(self):
		for i, name in enumerate(self.controlJoints):
			if i > 6:
				break

			self.joint = self.joints[name]

			pose1 = self.init_jointPositions[i]
			if i < 7:
				p.resetJointState(self.robotID, self.joint.id, targetValue=pose1, targetVelocity=0)
				
		p.resetBasePositionAndOrientation(self.laptopId, self.laptop_pos ,self.laptopStartOrientation)
		p.resetBasePositionAndOrientation(self.drawerId, self.drawer_pos ,self.drawerStartOrientation)

		p.resetJointState(self.laptopId, 0, targetValue= self.laptop_angle , targetVelocity=0)
		p.resetJointState(self.drawerId, 2, targetValue= self.drawer_joint1, targetVelocity=0)
		p.resetJointState(self.drawerId, 0, targetValue= self.drawer_joint2, targetVelocity=0)
	
		# p.stepSimulation()

		self.laptop_angle = p.getJointStates(self.laptopId, self.laptop_joint)[0][0]
		self.drawer_joint1 = p.getJointStates(self.drawerId, self.drawer_joint)[0][0]
		self.drawer_joint2 = p.getJointStates(self.drawerId, self.drawer_joint)[0][1]
		ee_position = np.array(p.getLinkState(self.robotID, self.eefID)[0])
		action_position = np.array(p.getLinkState(self.objectId[self.task_id], self.action_point_link[self.task_id])[0])
		self.dd_action = np.array(p.getLinkState(self.drawerId, 3)[0])
		penalty_position = np.array(p.getLinkState(self.objectId[self.task_id], self.penalty_point_link[self.task_id])[0])
		self.cur_variable = [self.laptop_angle, self.drawer_joint1]
		self.cur_dist = self.cal_distance(ee_position, action_position)
		self.cur_pen_dist = self.cal_distance(ee_position, penalty_position)
		self.ee_pos = np.array(p.getLinkState(self.robotID, self.eefID)[0])

		
	############################ Robot process ############################

	def step(self, action):

		self.move(action)
		self.next_state, task_ee_state, seg_img, dep_img, gt_state = self.get_state()
		self.steps += 1

		return self.next_state, self.reward, self.done, {}, task_ee_state, seg_img, dep_img, gt_state

	def move(self, action):

		ee_position = p.getLinkState(self.robotID, self.eefID)[0]
		ee_orientation = p.getLinkState(self.robotID, self.eefID)[1]
		ee_orientation = p.getEulerFromQuaternion(ee_orientation)

		if not self.stage1:
		 	p.setJointMotorControl2(self.laptopId, 0, p.VELOCITY_CONTROL, force=100)
		 	p.setJointMotorControl2(self.drawerId, 2, p.VELOCITY_CONTROL, force=10000)
		else:
		 	p.setJointMotorControl2(self.laptopId, 0, p.VELOCITY_CONTROL, force=5)
		 	p.setJointMotorControl2(self.drawerId, 2, p.VELOCITY_CONTROL, force=5)

		for j in range(12):
			target_position = []
			for i in range(3):
				target_position.append(ee_position[i]+(j+1)*0.02*action[i]/12.0)

			target_position[2] = max(target_position[2], 0.27)
			jointPose = p.calculateInverseKinematics(self.robotID, self.eefID, target_position, self.init_ee_ori)

			for i, name in enumerate(self.controlJoints):
				self.joint = self.joints[name]

				if i > 6:
					break
				pose1 = jointPose[i]
				if i < 7:  
					p.setJointMotorControl2(self.robotID, self.joint.id, p.POSITION_CONTROL,
											targetPosition=pose1, force=self.joint.maxForce, 
											maxVelocity=self.joint.maxVelocity/3.0)
			p.stepSimulation()


	############################ State process ############################


	def get_state(self):

		state, seg_img, dep_img = self.image_get()

		ee_states = p.getLinkState(self.robotID,  self.eefID, computeLinkVelocity = 1)
		ee_position = np.array(ee_states[0])
		ee_orientation = np.array(p.getEulerFromQuaternion(ee_states[1]))
		ee_linear_vel = np.array(ee_states[6])

		laptop_position = np.array(p.getLinkState(self.laptopId, 0)[0])
		laptop_orientation = np.array(p.getEulerFromQuaternion(p.getLinkState(self.laptopId, 0)[1]))

		drawer_position = np.array(p.getLinkState(self.drawerId, 0)[0])
		drawer_orientation = np.array(p.getEulerFromQuaternion(p.getLinkState(self.drawerId, 0)[1]))

		cur_laptop_angle = np.array(p.getJointStates(self.laptopId, self.laptop_joint)[0][0])
		cur_drawer_dist = np.array(p.getJointStates(self.drawerId, self.drawer_joint)[0][0])
		
		action_position = np.array(p.getLinkState(self.objectId[self.task_id], self.action_point_link[self.task_id])[0])

		rel_pos = action_position - ee_position
		rel_dist = self.cal_distance(ee_position, action_position)	

		if self.task_id == 0:
			self.done = self._is_success(cur_laptop_angle)
			penalty_position = np.array(p.getLinkState(self.laptopId, 2)[0])
			pen_rel_dist = self.cal_distance(ee_position, penalty_position)
			pen_rel_pos = penalty_position - ee_position
			self.reward = self.compute_total(cur_laptop_angle, rel_dist, pen_rel_dist)

		elif self.task_id == 1:
			self.done = self._is_success(cur_drawer_dist)
			penalty_position = np.array(p.getLinkState(self.drawerId, 3)[0])
			pen_rel_dist = self.cal_distance(ee_position, penalty_position)
			pen_rel_pos = penalty_position - ee_position
			self.reward = self.compute_total(cur_drawer_dist, rel_dist, pen_rel_dist)

		penalty_zone = np.array(0.0)
		if (rel_dist - pen_rel_dist) > 0.0:
			penalty_zone = np.array(1.0)

		task_ee_state = np.concatenate((self.task_state, self.task_state, ee_position, ee_linear_vel))
		gt_state = np.concatenate((ee_position, ee_linear_vel, rel_pos, pen_rel_pos, laptop_position, laptop_orientation, (cur_laptop_angle,), drawer_position, drawer_orientation, (cur_drawer_dist,), (penalty_zone,) , self.task_state))


		if self.task_id == 0 or (self.task_id == 1 and ee_position[2] <0.70):
			self.stage1 = True
		else:
			self.stage1 = False

		self.laptop_angle = cur_laptop_angle
		self.drawer_joint1 = cur_drawer_dist

		self.cur_variable = [self.laptop_angle, self.drawer_joint1]
		self.cur_dist = rel_dist
		self.cur_pen_dist = pen_rel_dist

		if self.done:
			print("OH YEAH")
			self.reward += 5

		return state, task_ee_state, seg_img, dep_img, gt_state

	def cal_distance(self, goal_a, goal_b):
		assert goal_a.shape == goal_b.shape
		return np.linalg.norm(goal_a - goal_b, axis=-1)

	def compute_total(self, cur_var, rel_dist, pen_dist):
		dif = -2.5*rel_dist + 0.8*pen_dist
		pre_dif = -2.5*self.cur_dist + 0.8*self.cur_pen_dist
		dist_reward = 100*(dif - pre_dif)


		if rel_dist > 0.04:
			dist_reward = -25*(rel_dist - self.cur_dist)
		else:
			dist_reward = 0.2

		task_reward = 10*(self.cur_variable[self.task_id] - cur_var)

		if self.task_id == 1:
			task_reward *= 8

		reward = 5*dist_reward + task_reward 

		return reward

	def _is_success(self, cur_var):
		done = (abs(cur_var) < self.threshold[self.task_id]).astype(np.float32)
		return done


	############################ Image process ############################
	
	def camera_setting(self):

		# Camera setting
		self.width = img_size
		self.height = img_size
		self.view_matrix = p.computeViewMatrix([-0.027, 0.436, 0.939], [1.288, -0.086, 0.468], [1.19, 0.00, 0.992])
		self.fov = 59
		self.aspect = self.width / self.height
		self.aspect = 1280.0/720.0
		self.near = 0.1
		self.far = 1.5
		self.projection_matrix = p.computeProjectionMatrixFOV(self.fov, self.aspect, self.near, self.far)
		
	def image_get(self):
		
		direction = [ -3.5, -0.3, 3.0]
		images = p.getCameraImage(self.width,self.height,self.view_matrix,self.projection_matrix, shadow=True, lightColor=(0.8, 0.8, 0.8), lightDirection=direction,lightAmbientCoeff=0.5, lightDiffuseCoeff=0.5, lightSpecularCoeff=0.02)

		rgb = np.reshape(images[2], (self.height, self.width, 4))
		seg_img, dep_img = self.seg_dep_get(images[4], images[3])
		ori_img = rgb[:,:,:3]

		
		# ori_img = self.gau_noisy(ori_img)
		# dd = 255*seg_img[:,:,:3] + 100*seg_img[:,:,3:]
		# cc = np.concatenate((dep_img,dep_img,dep_img),axis=2)
		# con_img = np.concatenate((ori_img,dd,cc),axis=1)
		# plt.imsave(f'sample/ori/{self.epi}_{self.steps}_ori.png',con_img.astype(np.uint8))
		ori_img, seg_img, dep_img =  np.transpose(ori_img,[2, 0, 1]), np.transpose(seg_img,[2, 0, 1]), np.transpose(dep_img,[2, 0, 1])

		return ori_img, seg_img, dep_img

	def seg_dep_get(self, seg, dep):
		seg_img = np.zeros((self.height, self.width, SEG_NUM))

		for ind in self.robot_index:
			seg_img[seg == ind,0] = 255

		for ind in self.table_index:
			seg_img[seg == ind,1] = 255
		
		for ind in self.wall_index:
			seg_img[seg == ind,2] = 255
		
		for ind in self.laptop_index:
			seg_img[seg == ind,3] = 255
			
		for ind in self.drawer_index:
			seg_img[seg == ind,4] = 255
			
		for ind in self.obj_index:
			seg_img[seg == ind,5] = 255
			
		dep_img = np.reshape(dep,(self.height,self.width))
		min_dep = MIN_DEP
		max_dep = MAX_DEP
		dep_img = 255*((dep_img-min_dep)/(max_dep - min_dep))
		dep_img = np.reshape(dep_img, (self.height, self.width, 1))

		return seg_img, dep_img

	def index(self):
		self.robot_index = []
		self.table_index = []
		self.wall_index = []
		self.laptop_index = []
		self.drawer_index = []
		self.obj_index = []

		for i in range(21):
			self.robot_index.append(self.robotID+((i)<<24))

		for i in range(2):
			self.table_index.append(self.tableID+((i)<<24))

		for i in range(2):
			self.wall_index.append(self.wall1Id+((i)<<24))
			self.wall_index.append(self.wall2Id+((i)<<24))
			self.wall_index.append(self.wall3Id+((i)<<24))

		for i in range(3):
			self.laptop_index.append([self.laptopId+(i<<24)])

		for i in range(16):
			self.drawer_index.append([self.drawerId+(i<<24)])

		self.obj_index.append(self.lotionId+((0)<<24))
		self.obj_index.append(self.bottleId+((0)<<24))
		self.obj_index.append(self.bottle_capId+((0)<<24))

		for i in range(6):			
			self.obj_index.append(self.binId+((i)<<24))
			self.obj_index.append(self.bin_coverId+((i)<<24))

	def gau_noisy(self, image):
		eps = random.uniform(0,1)

		if eps < 0.5 :
			row,col,ch= image.shape
			mean = 0
			var = 50
			sigma = var**0.5
			gauss = np.random.normal(mean,sigma,(row,col,ch))
			gauss = gauss.reshape(row,col,ch)
			noisy = image + gauss
			noisy = np.clip(noisy,0,255)
			return noisy

		else:
			return image

	def set_color(self):


		# change = np.random.uniform(-0.03,0.03,4)
		# change[3] = 0
		# for i in range(9):
		# 	color = np.array([0.9,0.9,0.9,1])
		# 	if i == 8 or i ==7:
		# 		color = np.array([0.5647,0.6118,0.6249,1])
		# 	elif i ==6:
		# 		color = np.array([0.7647,0.8118,0.8249,1])
		# 	color += change
		# 	p.changeVisualShape(self.robotID, i, rgbaColor=color)

		# change = np.random.uniform(-0.05,0.05,4)
		# change[3] = 0
		# for i in range(9,25):
		# 	color = np.array([0.2,0.2,0.25,1])
		# 	if i == 14 or i == 12:
		# 		color = np.array([0.8,0.8,0.8,1])
		# 	elif i == self.eefID:
		# 		color = np.array([0.7,0.1,0.1,1])			
		# 	color += change
		# 	p.changeVisualShape(self.robotID, i, rgbaColor=color)

		# change = np.random.uniform(-0.03,0.03,4)
		# change[3] = 0
		# for i in range(1):
		# 	color = np.array([0.97,0.97,0.97,1])
		# 	color += change
		# 	p.changeVisualShape(self.wall1Id,-1, rgbaColor=color)
		# 	p.changeVisualShape(self.wall2Id,-1, rgbaColor=color)
		# 	p.changeVisualShape(self.wall3Id,-1, rgbaColor=color)

		# p.changeVisualShape(self.binId,-1, rgbaColor=color)

		# change = np.random.uniform(-0.05,0.05,4)
		# change[3] = 0
		# for i in range(3):
		# 	color = np.array([0.9,0.9,0.9,1])
		# 	color += change
		# 	p.changeVisualShape(self.laptopId,i, rgbaColor=color)
		# p.changeVisualShape(self.laptopId,-1, rgbaColor=color)

		# change = np.random.uniform(-0.03,0.03,4)
		# change[3] = 0
		# for i in range(6):
		# 	color = ([0.97,0.97,0.97,1])
		# 	color += change
		# 	p.changeVisualShape(self.drawerId,i, rgbaColor=color)
		# p.changeVisualShape(self.drawerId,-1, rgbaColor=color)

		change = np.random.uniform(-0.03,0.03,4)
		change[3] = 0
		for i in range(1):
			color = ([1.0,0.9,0.7,1])
			color += change
			p.changeVisualShape(self.tableID,-1, rgbaColor=color)


		change = np.random.uniform(-0.03,0.03,4)
		change[3] = 0
		for i in range(1):
			color = ([0.97,0.97,0.97,1])
			color += change
			p.changeVisualShape(self.binId,-1, rgbaColor=color)
			p.changeVisualShape(self.bin_coverId,-1, rgbaColor=color)

		change = np.random.uniform(-0.04,0.04,4)
		change[3] = 0
		for i in range(1):
			color = ([0.95,0.95,0.95,1])
			color += change
			p.changeVisualShape(self.lotionId,-1, rgbaColor=color)

		change = np.random.uniform(-0.04,0.04,4)
		change[3] = 0
		for i in range(1):
			color = ([0.95,0.95,0.95,1])
			color += change
			p.changeVisualShape(self.bottleId,-1, rgbaColor=color)

		change = np.random.uniform(-0.04,0.04,4)
		change[3] = 0
		for i in range(1):
			color = ([0.95,0.95,0.95,1])
			color += change
			p.changeVisualShape(self.bottle_capId,-1, rgbaColor=color)

	def resetObject(self):

		# bin_body random
		self.bin_body_pos = [2.0, 2.0, 2.0]
		if (0 in self.obj_list):
			while True:				
				self.bin_body_ori = random.uniform(0.0, 3.141592)
				self.bin_body_ori = p.getQuaternionFromEuler([0 ,0, self.bin_body_ori])
				self.bin_body_pos[2] = 0.544+0.0015
				self.bin_body_pos[0] = random.uniform(0.42, 0.75)
				self.bin_body_pos[1] = random.uniform(-0.475,0.475)
				# print("1")
				self.trial +=1 
				if self.trial > 100:
					self.reset()
				if (self.cal_distance(np.array(self.laptop_pos), np.array(self.bin_body_pos)) > 0.31 and
					self.cal_distance(np.array(self.drawer_pos), np.array(self.bin_body_pos)) > 0.24):
					break
			p.resetBasePositionAndOrientation(self.binId, self.bin_body_pos ,self.bin_body_ori)

		# bin_cover random
		self.bin_cover_pos = [2.0, 2.0, 2.0]
		if (1 in self.obj_list):
			eps = random.uniform(0,1)			
			if eps < 0.3 and (0 in self.obj_list):
				self.bin_cover_pos = self.bin_body_pos.copy()
				self.bin_cover_pos[2] += 0.055
				self.bin_cover_ori = self.bin_body_ori
			else:
				while True:			
					self.bin_cover_ori = random.uniform(0.0, 3.141592)
					self.bin_cover_ori = p.getQuaternionFromEuler([0 ,0, self.bin_cover_ori])
					self.bin_cover_pos[2] = 0.544
					self.bin_cover_pos[0] = random.uniform(0.42, 0.75)
					self.bin_cover_pos[1] = random.uniform(-0.475,0.475)
					# print("2")
					self.trial +=1 
					if self.trial > 100:
						self.reset()
					if (self.cal_distance(np.array(self.laptop_pos), np.array(self.bin_cover_pos)) > 0.31 and
					self.cal_distance(np.array(self.drawer_pos), np.array(self.bin_cover_pos)) > 0.22 and
					self.cal_distance(np.array(self.bin_body_pos), np.array(self.bin_cover_pos)) > 0.12):
						break


			p.resetBasePositionAndOrientation(self.bin_coverId, self.bin_cover_pos ,self.bin_cover_ori)

		# bottle body random
		self.bottle_body_pos = [2.0, 2.0, 2.0]
		bottle_body = 0
		if (2 in self.obj_list):
			while True:
				bottle_body = random.randint(0, 1)				
				self.bottle_body_ori = random.uniform(0.0, 3.141592)
				if bottle_body == 0:
					self.bottle_body_ori = p.getQuaternionFromEuler([0 ,0, self.bottle_body_ori])
					self.bottle_body_pos[2] = 0.544+0.033
				elif bottle_body == 1:
					
					self.bottle_body_ori = p.getQuaternionFromEuler([3.141592/2.0 ,0, self.bottle_body_ori])
					self.bottle_body_pos[2] = 0.544		
				self.bottle_body_pos[0] = random.uniform(0.42, 0.75)
				self.bottle_body_pos[1] = random.uniform(-0.45,0.45)
				# print("3")
				self.trial +=1 
				if self.trial > 200:
					self.reset()
				if (self.cal_distance(np.array(self.laptop_pos), np.array(self.bottle_body_pos)) > 0.30 and
					self.cal_distance(np.array(self.drawer_pos), np.array(self.bottle_body_pos)) > 0.22 and
					self.cal_distance(np.array(self.dd_action), np.array(self.bottle_body_pos)) > 0.22 and
					self.cal_distance(np.array(self.bin_body_pos), np.array(self.bottle_body_pos)) > 0.13 and 
					self.cal_distance(np.array(self.bin_cover_pos), np.array(self.bottle_body_pos)) > 0.12):
					break

			
			p.resetBasePositionAndOrientation(self.bottleId, self.bottle_body_pos ,self.bottle_body_ori)

		# bottle_cap random
		self.bottle_cap_pos = [2.0, 2.0, 2.0]
		if (3 in self.obj_list):
			eps = random.uniform(0,1)			
			if eps < 0.6 and bottle_body == 1 and (2 in self.obj_list):
				self.bottle_cap_pos = self.bottle_body_pos.copy()
				self.bottle_cap_pos[2] += 0.11
				self.bottle_cap_ori = random.uniform(0.0, 3.141592)
				self.bottle_cap_ori = p.getQuaternionFromEuler([0 ,0, self.bottle_cap_ori])
			else:
				while True:			
					self.bottle_cap_ori = random.uniform(0.0, 3.141592)
					self.bottle_cap_ori = p.getQuaternionFromEuler([0 ,0, self.bottle_cap_ori])
					self.bottle_cap_pos[2] = 0.544
					self.bottle_cap_pos[0] = random.uniform(0.38, 0.84)
					self.bottle_cap_pos[1] = random.uniform(-0.55,0.55)
					# print("4")
					self.trial +=1 
					if self.trial > 200:
						self.reset()
					if (self.cal_distance(np.array(self.laptop_pos), np.array(self.bottle_cap_pos)) > 0.27 and
					self.cal_distance(np.array(self.drawer_pos), np.array(self.bottle_cap_pos)) > 0.235 and
					self.cal_distance(np.array(self.bin_body_pos), np.array(self.bottle_cap_pos)) > 0.15 and
					self.cal_distance(np.array(self.bin_cover_pos), np.array(self.bottle_cap_pos))> 0.1 and
					self.cal_distance(np.array(self.bottle_body_pos), np.array(self.bottle_cap_pos)) >0.13):
						break

			p.resetBasePositionAndOrientation(self.bottle_capId, self.bottle_cap_pos ,self.bottle_cap_ori)

		# lotion random
		self.lotion_pos = [2.0, 2.0, 2.0]
		if (4 in self.obj_list):
			while True:
				lotion = random.randint(0, 1)				
				self.lotion_ori = random.uniform(0.0, 3.141592)
				if lotion == 0:
					self.lotion_ori = p.getQuaternionFromEuler([-0.2 ,0, self.lotion_ori])
					self.lotion_pos[2] = 0.544+0.02
				else:			
					self.lotion_ori = p.getQuaternionFromEuler([3.141592/2.0,0, self.lotion_ori])
					self.lotion_pos[2] = 0.544+0.019		
				self.lotion_pos[0] = random.uniform(0.42, 0.78)
				self.lotion_pos[1] = random.uniform(-0.49,0.49)
				# print("5")
				self.trial +=1 
				if self.trial > 300:
					self.reset()
				if (self.cal_distance(np.array(self.laptop_pos), np.array(self.lotion_pos)) > 0.26 and
					self.cal_distance(np.array(self.drawer_pos), np.array(self.lotion_pos)) > 0.235 and
					self.cal_distance(np.array(self.bin_body_pos), np.array(self.lotion_pos)) > 0.15 and
					self.cal_distance(np.array(self.bin_cover_pos), np.array(self.lotion_pos))> 0.1 and
					self.cal_distance(np.array(self.bottle_body_pos), np.array(self.lotion_pos)) >0.1 and
					self.cal_distance(np.array(self.bottle_cap_pos), np.array(self.lotion_pos)) >0.1):
						break

			p.resetBasePositionAndOrientation(self.lotionId, self.lotion_pos ,self.lotion_ori)

		p.stepSimulation()

	def reload(self):
		p.removeBody(self.drawerId)

		size = random.uniform(0.90,1.03)

		drawerStartPos = [0.68, 0.2, 0.57]
		drawerStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
		self.drawerId = p.loadURDF("./urdf/objects/drawer.urdf", drawerStartPos, drawerStartOrientation,globalScaling=size,useFixedBase = True,flags=p.URDF_USE_SELF_COLLISION)

		self.objectId = [self.laptopId, self.drawerId]

	def canonical_color(self):
		
		self.ROBOT_COLOR = [[0, 0, 0.9, 1], [0, 0.9, 0, 1], [0.9, 0, 0, 1], [0.9, 0.9, 0, 1]]
		self.TABLE_COLOR = [[0.937, 0.729, 0.494, 1]]
		self.WALL_COLOR = [[0.82, 0.82, 0.82, 1]]

		self.LAPTOP_COLOR = [[0.0, 0.4196, 0.502, 1]]
		self.DRAWER_COLOR = [[0.9, 0.10, 0.68, 1]]

		for i in range(25):
			color = self.ROBOT_COLOR[i%len(self.ROBOT_COLOR)]
			p.changeVisualShape(self.robotID, i, rgbaColor=color)

		for i in range(1):
			color = self.WALL_COLOR[i]
			p.changeVisualShape(self.wall1Id,-1, rgbaColor=color)
			p.changeVisualShape(self.wall2Id,-1, rgbaColor=color)
			p.changeVisualShape(self.wall3Id,-1, rgbaColor=color)

		for i in range(1):
			color = self.TABLE_COLOR[i]
			p.changeVisualShape(self.tableID,-1, rgbaColor=color)

		for i in range(3):
			color = self.LAPTOP_COLOR[0]
			p.changeVisualShape(self.laptopId, i, rgbaColor=color)
		p.changeVisualShape(self.laptopId,-1, rgbaColor=color)

		for i in range(10):
			color = self.DRAWER_COLOR[0]
			p.changeVisualShape(self.drawerId,i, rgbaColor=color)
		p.changeVisualShape(self.drawerId,-1, rgbaColor=color)

	def selectObject(self):

		self.num_obj = random.randint(0,5)
		self.obj_list = random.sample([0,1,2,3,4], self.num_obj)

		self.removeObject()
		self.resetObject()

	def removeObject(self):
		for i, obj_id in enumerate(self.obj_id_list):
			position = [0, i+1, 0.0]
			orientation = [0,0,0,0]
			p.resetBasePositionAndOrientation(obj_id, position ,orientation)

	def get_num_obj(self):
		return self.num_obj

	def seed(self, seed):
		random.seed(seed)
