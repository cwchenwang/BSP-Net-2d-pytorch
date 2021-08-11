import os
import time
import math
import random
import numpy as np
import h5py
import cv2

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

from bspt_2d import *

class encoder(nn.Module):
	def __init__(self, ef_dim, p_dim):
		super(encoder, self).__init__()
		self.ef_dim = ef_dim
		self.p_dim = p_dim
		self.conv_1 = nn.Conv2d(1, self.ef_dim, 4, stride=2, padding=1, bias=True)
		self.conv_2 = nn.Conv2d(self.ef_dim, self.ef_dim*2, 4, stride=2, padding=1, bias=True)
		self.conv_3 = nn.Conv2d(self.ef_dim*2, self.ef_dim*4, 4, stride=2, padding=1, bias=True)
		self.conv_4 = nn.Conv2d(self.ef_dim*4, self.ef_dim*8, 4, stride=2, padding=1, bias=True)
		self.conv_5 = nn.Conv2d(self.ef_dim*8, self.ef_dim*8, 4, stride=1, padding=0, bias=True)
		nn.init.xavier_uniform_(self.conv_1.weight)
		nn.init.constant_(self.conv_1.bias,0)
		nn.init.xavier_uniform_(self.conv_2.weight)
		nn.init.constant_(self.conv_2.bias,0)
		nn.init.xavier_uniform_(self.conv_3.weight)
		nn.init.constant_(self.conv_3.bias,0)
		nn.init.xavier_uniform_(self.conv_4.weight)
		nn.init.constant_(self.conv_4.bias,0)
		nn.init.xavier_uniform_(self.conv_5.weight)
		nn.init.constant_(self.conv_5.bias,0)

		self.l1 = nn.Linear(self.ef_dim*8, self.ef_dim*16)
		self.l2 = nn.Linear(self.ef_dim*16, self.ef_dim*32)
		self.l3 = nn.Linear(self.ef_dim*32, self.ef_dim*64)
		self.l4_m = nn.Linear(self.ef_dim*64, self.p_dim*2)
		self.l4_b = nn.Linear(self.ef_dim*64, self.p_dim)
		nn.init.xavier_uniform_(self.l1.weight)
		nn.init.constant_(self.l1.bias,0)
		nn.init.xavier_uniform_(self.l2.weight)
		nn.init.constant_(self.l2.bias,0)
		nn.init.xavier_uniform_(self.l3.weight)
		nn.init.constant_(self.l3.bias,0)
		nn.init.xavier_uniform_(self.l4_m.weight)
		nn.init.constant_(self.l4_m.bias,0)
		nn.init.xavier_uniform_(self.l4_b.weight)
		nn.init.constant_(self.l4_b.bias,0)

	def forward(self, inputs, is_training=False):
		d_1 = self.conv_1(inputs)
		d_1 = F.leaky_relu(d_1, negative_slope=0.01, inplace=True)

		d_2 = self.conv_2(d_1)
		d_2 = F.leaky_relu(d_2, negative_slope=0.01, inplace=True)
		
		d_3 = self.conv_3(d_2)
		d_3 = F.leaky_relu(d_3, negative_slope=0.01, inplace=True)

		d_4 = self.conv_4(d_3)
		d_4 = F.leaky_relu(d_4, negative_slope=0.01, inplace=True)

		d_5 = self.conv_5(d_4)
		d_5 = F.leaky_relu(d_5, negative_slope=0.01, inplace=True)
		d_5 = d_5.view(-1, self.ef_dim*8)
		# d_5 = torch.sigmoid(d_5)

		# decode
		l_1 = self.l1(d_5)
		l_1 = F.leaky_relu(l_1, negative_slope=0.01, inplace=True)

		l_2 = self.l2(l_1)
		l_2 = F.leaky_relu(l_2, negative_slope=0.01, inplace=True)

		l_3 = self.l3(l_2)
		l_3 = F.leaky_relu(l_3, negative_slope=0.01, inplace=True)

		l_4m = self.l4_m(l_3)
		l_4b = self.l4_b(l_3)
		
		l_4m = l_4m.view(-1, 2, self.p_dim)
		l_4b = l_4b.view(-1, 1, self.p_dim)
		# print("encoder shape", l_4m.shape, l_4b.shape)

		return l_4m, l_4b
		
class generator(nn.Module):
	def __init__(self, phase, p_dim, c_dim):
		super(generator, self).__init__()
		self.phase = phase
		self.p_dim = p_dim
		self.c_dim = c_dim
		convex_layer_weights = torch.zeros((self.p_dim, self.c_dim))
		concave_layer_weights = torch.zeros((self.c_dim, 1))
		self.convex_layer_weights = nn.Parameter(convex_layer_weights)
		self.concave_layer_weights = nn.Parameter(concave_layer_weights)
		nn.init.normal_(self.convex_layer_weights, mean=0.0, std=0.02)
		nn.init.normal_(self.concave_layer_weights, mean=1e-5, std=0.02)
	
	def forward(self, points, plane_m, plane_b, is_training=False):
		# print('point shape', points.shape)
		# print(plane_m.shape, plane_b.shape)
		if self.phase == 0:
			h1 = torch.matmul(points, plane_m) + plane_b
			h1 = torch.clamp(h1, min=0)

			#level 2
			h2 = torch.matmul(h1, self.convex_layer_weights)
			h2 = torch.clamp(1-h2, min=0, max=1)

			#level 3
			h3 = torch.matmul(h2, self.concave_layer_weights)
			h3 = torch.clamp(h3, min=0, max=1)
			h3_max, _ = torch.max(h2, 2, keepdims=True)
			return h3, h3_max, h2
		elif self.phase == 1 or self.phase == 2:
			h1 = torch.matmul(points, plane_m) + plane_b
			h1 = torch.clamp(h1, min=0)
			#level 2
			h2 = torch.matmul(h1, (self.convex_layer_weights>0.01).float())

			#level 3
			h3 = torch.min(h2, dim=2, keepdim=True)[0]
			# h3_01, _ = torch.max(torch.min(1-h3, 1), 0)
			h3_01 = torch.clamp(1-h3, min=0, max=1)
			# h3_01 = 0

			return h3, h3_01, h2
		

class bsp_network(nn.Module):
	def __init__(self, phase, ef_dim, p_dim, c_dim):
		super(bsp_network, self).__init__()
		self.phase = phase
		self.ef_dim = ef_dim
		self.p_dim = p_dim # num of planes
		self.c_dim = c_dim # num of convexes
		self.encoder = encoder(self.ef_dim, self.p_dim)
		self.generator = generator(self.phase, self.p_dim, self.c_dim)

	def forward(self, inputs, z_vector, plane_m, point_coord, convex_mask=None, is_training=False):
		if is_training:
			plane_m, plane_b = self.encoder(inputs, is_training=True)
			# TODO: convex_mask ?
			G, G_max, G2 = self.generator(point_coord, plane_m, plane_b, is_training=True)
			# print('G shape', G.shape, G2.shape)
		else:
			if inputs is not None:
				plane_m, plane_b = self.encoder(inputs, is_training=is_training)
			if point_coord is not None:
				G, G_max, G2 = self.generator(point_coord, plane_m, plane_b, is_training=is_training)
			# if inputs is not None:
			# 	z_vector = self.encoder(inputs, is_training=is_training)
			# if z_vector is not None:
			# 	plane_m = self.decoder(z_vector, is_training=is_training)
			# if point_coord is not None:
			# 	net_out_convexes, net_out = self.generator(point_coord, plane_m, convex_mask=convex_mask, is_training=is_training)
			# else:
			# 	net_out_convexes = None
			# 	net_out = None
			# return z_vector, plane_m, net_out_convexes, net_out

		return G, G_max, G2, plane_m, plane_b

class IMSEG(object):
	def __init__(self, config, shape_batch_size=8, ef_dim=32, c_dim=256, p_dim=256):
		self.phase = config.phase
		self.sample_vox_size = config.sample_vox_size
		self.point_batch_size = self.sample_vox_size*self.sample_vox_size
		self.shape_batch_size = shape_batch_size # 24

		self.p_dim = p_dim
		self.ef_dim = ef_dim
		self.c_dim = c_dim

		self.dataset_name = config.dataset
		self.checkpoint_dir = config.checkpoint_dir
		self.data_dir = config.data_dir
		
		data_hdf5_name = self.data_dir+'/'+self.dataset_name+'.hdf5'
		if os.path.exists(data_hdf5_name):
			self.data_dict = h5py.File(data_hdf5_name, 'r')
			self.data_voxels = self.data_dict['pixels'][:]
		else:
			print("error: cannot load "+data_hdf5_name)
			exit(0)
		dim = self.sample_vox_size
		self.coords = np.zeros([dim,dim,2],np.float32)
		for i in range(dim):
			for j in range(dim):
				self.coords[i,j,0] = i
				self.coords[i,j,1] = j
		self.coords = (self.coords+0.5)/dim-0.5
		self.coords = np.tile(np.reshape(self.coords,[1,self.point_batch_size,2]),[self.shape_batch_size,1,1])
		self.coords = torch.from_numpy(self.coords)

		if torch.cuda.is_available():
			self.device = torch.device('cuda')
			torch.backends.cudnn.benchmark = True
		else:
			self.device = torch.device('cpu')
		self.coords = self.coords.to(self.device)
		

		self.bsp_network = bsp_network(self.phase, self.ef_dim, self.p_dim, self.c_dim)
		self.bsp_network.to(self.device)
		self.optimizer = torch.optim.Adam(self.bsp_network.parameters(), lr=config.learning_rate, betas=(config.beta1, 0.999))

		self.max_to_keep = 2
		self.checkpoint_path = os.path.join(self.checkpoint_dir, self.model_dir)
		self.checkpoint_name='BSP_AE.model'
		self.checkpoint_manager_list = [None] * self.max_to_keep
		self.checkpoint_manager_pointer = 0

		if config.phase == 0:
			def network_loss(G2,G,point_value,cw2,cw3):
				loss_sp = torch.mean((point_value - G)**2)
				loss_t =  torch.sum(torch.abs(cw3-1)) + (torch.sum(torch.clamp(cw2-1, min=0) - torch.clamp(cw2, max=0)))
				loss = loss_sp + loss_t
				return loss_sp, loss
			self.loss = network_loss
		elif config.phase==1:
			#phase 1 hard discrete for bsp
			#L_recon
			def network_loss(G2,G,point_value,cw2,cw3):
				loss_sp = torch.mean((1-point_value)*(1-torch.clamp(G, max=1)) + point_value*(torch.clamp(G, min=0)))
				loss = loss_sp
				return loss_sp,loss
			self.loss = network_loss
		elif config.phase==2:
			#phase 2 hard discrete for bsp with L_overlap
			#L_recon + L_overlap
			def network_loss(G2,G,point_value,cw2,cw3):
				loss_sp = torch.mean((1-point_value)*(1-torch.clamp(G, max=1)) + point_value*(torch.clamp(G, min=0)))
				G2_inside = (G2<0.01).float()
				bmask = G2_inside * (torch.sum(G2_inside, dim=2, keepdim=True)>1).float()
				bmask = torch.mean(G2*point_value*bmask)
				loss = loss_sp - bmask
				return loss_sp, loss
			self.loss = network_loss

	@property
	def model_dir(self):
		return "{}_ae".format(self.dataset_name)

	def load(self):
		# Debug: load directly
		# self.bsp_network.load_state_dict(torch.load('checkpoint/complex_elements_ae/bsp_tf_pre_p2.pt'))
		checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
		if os.path.exists(checkpoint_txt):
			fin = open(checkpoint_txt)
			model_dir = fin.readline().strip()
			fin.close()
			self.bsp_network.load_state_dict(torch.load(model_dir))
			print(" [*] Load SUCCESS with model: {}".format(model_dir))
			return True
		else:
			print(" [!] Load failed...")
			return False

	def save(self,epoch):
		if not os.path.exists(self.checkpoint_path):
			os.makedirs(self.checkpoint_path)
		save_dir = os.path.join(self.checkpoint_path,self.checkpoint_name+str(self.sample_vox_size)+"-"+str(self.phase)+"-"+str(epoch)+".pth")
		self.checkpoint_manager_pointer = (self.checkpoint_manager_pointer+1)%self.max_to_keep
		#delete checkpoint
		if self.checkpoint_manager_list[self.checkpoint_manager_pointer] is not None:
			if os.path.exists(self.checkpoint_manager_list[self.checkpoint_manager_pointer]):
				os.remove(self.checkpoint_manager_list[self.checkpoint_manager_pointer])
		#save checkpoint
		torch.save(self.bsp_network.state_dict(), save_dir)
		#update checkpoint manager
		self.checkpoint_manager_list[self.checkpoint_manager_pointer] = save_dir
		#write file
		checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
		fout = open(checkpoint_txt, 'w')
		for i in range(self.max_to_keep):
			pointer = (self.checkpoint_manager_pointer+self.max_to_keep-i)%self.max_to_keep
			if self.checkpoint_manager_list[pointer] is not None:
				fout.write(self.checkpoint_manager_list[pointer]+"\n")
		fout.close()

	def train(self, config):
		self.load()
		
		# self.test_1(config,"train_"+str(self.sample_vox_size))
		# return

		shape_num = len(self.data_voxels)
		batch_index_list = np.arange(shape_num)
		
		print("\n\n----------net summary----------")
		print("training samples   ", shape_num)
		print("-------------------------------\n\n")	
		start_time = time.time()
		assert config.epoch==0 or config.iteration==0
		training_epoch = config.epoch + int(config.iteration/shape_num)
		batch_num = int(shape_num/self.shape_batch_size)
		# point_batch_num = int(self.load_point_batch_size/self.point_batch_size)
		self.bsp_network.train()
		for epoch in range(0, training_epoch):
			# TODO: for debug
			# np.random.shuffle(batch_index_list)
			avg_loss_sp = 0
			avg_loss_tt = 0
			avg_num = 0
			for idx in range(batch_num):
				dxb = batch_index_list[idx*self.shape_batch_size:(idx+1)*self.shape_batch_size]
				batch_voxels = self.data_voxels[dxb].astype(np.float32)

				batch_voxels = torch.from_numpy(batch_voxels).permute(0, 3, 1, 2)
				point_coord = self.coords.clone().detach()
				point_value = batch_voxels.clone().reshape(self.shape_batch_size, self.point_batch_size, 1)

				# print(batch_voxels.shape, point_coord.shape, point_value.shape)

				batch_voxels = batch_voxels.to(self.device)
				point_coord = point_coord.to(self.device)
				point_value = point_value.to(self.device)

				self.bsp_network.zero_grad()
				G, G_max, G2, _, _ = self.bsp_network(batch_voxels, None, None, point_coord, is_training=True)
				# print(G.shape)
				#errSP, errTT, errM = self.loss(G2, G, point_value, self.bsp_network.generator.convex_layer_weights, None)
				errSP, errTT = self.loss(G2, G, point_value, self.bsp_network.generator.convex_layer_weights, self.bsp_network.generator.concave_layer_weights)

				errTT.backward()
				self.optimizer.step()

				avg_loss_sp += errSP.item()
				avg_loss_tt += errTT.item()
				avg_num += 1

				# if(epoch == 0 and idx == 0):
				# 	np.savetxt('G.out', G.cpu().detach().numpy()[0].reshape(64, 64), delimiter=',')
					# np.savetxt('P.out', point_coord.cpu().detach().numpy().reshape(20, 64, 64), delimiter=',')
				#print(str(self.sample_vox_size)+" Epoch: [%2d/%2d] time: %4.4f, loss_sp: %.6f, loss_t: %.6f, loss_total: %.6f" % (epoch, training_epoch, time.time() - start_time, errSP.item(), errT.item(), errTT.item()))
				#self.test_1(config,"train_"+str(self.sample_vox_size)+"_"+str(epoch))
			print(str(self.sample_vox_size)+" Epoch: [%2d/%2d] time: %4.4f, loss_sp: %.8f, loss_total: %.8f" % (epoch, training_epoch, time.time() - start_time, avg_loss_sp/avg_num, avg_loss_tt/avg_num))
			if epoch%20==1:
				#print("testing")
				self.test_1(config,"train_"+str(self.sample_vox_size)+"_"+str(epoch))
			if epoch % config.save_freq == 1:
				self.save(epoch)

		self.save(training_epoch)

	def test_1(self, config, name):
		t = 0
		batch_voxels = self.data_voxels[t:t+self.shape_batch_size].astype(np.float32)
		batch_voxels = torch.from_numpy(batch_voxels).permute(0, 3, 1, 2).to(self.device)
		point_coord = self.coords.clone().detach()
		point_coord = point_coord.to(self.device)
		model_out, model_out1, _, _, _ = self.bsp_network(batch_voxels, None, None, point_coord, is_training=False)
		if config.phase == 1 or config.phase == 2:
			model_out = model_out1

		imgs = np.clip(np.resize(model_out.detach().cpu().numpy(),
			[self.shape_batch_size,self.sample_vox_size,self.sample_vox_size])*256, 0, 255).astype(np.uint8)
		np.savetxt('input.out', batch_voxels.detach().cpu().numpy()[0].reshape(64, 64), fmt='%.0f', delimiter=',')
		for t in range(self.shape_batch_size):
			cv2.imwrite("{}/{}_out.png".format(config.sample_dir, t), imgs[t])
			cv2.imwrite("{}/{}_gt.png".format(config.sample_dir, t), batch_voxels[t].permute(1, 2, 0).detach().cpu().numpy()*255)

		if config.phase==1 or config.phase==2:
			image_out_size = 256 # size for BSP illustration
			w2 = self.bsp_network.generator.convex_layer_weights.detach().cpu().numpy()

			start_n = config.start
			batch_voxels = self.data_voxels[start_n:start_n+self.shape_batch_size].astype(np.float32)
			batch_voxels = torch.from_numpy(batch_voxels).permute(0, 3, 1, 2).to(self.device)
			point_coord = self.coords.clone().detach()
			point_coord = point_coord.to(self.device)
			_, _, model_out, out_m, out_b = self.bsp_network(batch_voxels, None, None, point_coord, is_training=False)

			# print(model_out.detach().cpu().numpy().shape)
			# np.savetxt('model_out.out', model_out.detach().cpu().numpy()[0], delimiter=',')

			model_out = np.resize(model_out.detach().cpu().numpy(),
			[self.shape_batch_size,self.sample_vox_size,self.sample_vox_size,self.c_dim])

			# np.savetxt('input.out', batch_voxels.detach().cpu().numpy()[0].reshape(64, 64), fmt='%.3e', delimiter=',')
			# print(model_out.shape)
			# np.savetxt('m_out.out', out_m.detach().cpu().numpy()[0], delimiter=',')
			# np.savetxt('b_out.out', out_b.detach().cpu().numpy()[0], delimiter=',')

			for t in range(self.shape_batch_size):
				bsp_convex_list = []
				color_list = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]
				color_idx_list = []

				for i in range(self.c_dim):
					min_v = np.min(model_out[t,:,:,i])
					if min_v<0.01:
						box = []
						for j in range(self.p_dim):
							if w2[j,i]>0.01:
								a = -out_m[t,0,j]
								b = -out_m[t,1,j]
								d = -out_b[t,0,j]
								box.append([a,b,d])
						if len(box)>0:
							bsp_convex_list.append(np.array(box,np.float32))
							color_idx_list.append(i%len(color_list))

				#print(bsp_convex_list)
				# print(len(bsp_convex_list))
				
				#convert bspt to mesh
				vertices = []
				polygons = []
				polygons_color = []

				img_out = np.full([image_out_size,image_out_size,3],255,np.uint8)
				for i in range(len(bsp_convex_list)):
					vg, tg = digest_bsp(bsp_convex_list[i], bias=0)
					cg = color_list[color_idx_list[i]]
					for j in range(len(tg)):
						x1 = ((vg[tg[j][0]][1]+0.5)*image_out_size).astype(np.int32)
						y1 = ((vg[tg[j][0]][0]+0.5)*image_out_size).astype(np.int32)
						x2 = ((vg[tg[j][1]][1]+0.5)*image_out_size).astype(np.int32)
						y2 = ((vg[tg[j][1]][0]+0.5)*image_out_size).astype(np.int32)
						cv2.line(img_out, (x1,y1), (x2,y2), cg, thickness=1)
				
				cv2.imwrite(config.sample_dir+"/"+str(t)+"_bsp.png", img_out)
	
	def test_bsp(self, config):
		if not self.load():
			exit(-1)
		
		w2 = self.bsp_network.generator.convex_layer_weights.detach().cpu().numpy()
		self.bsp_network.eval()
		start_n = config.start
		batch_voxels = self.data_voxels[start_n:start_n+self.shape_batch_size].astype(np.float32)
		batch_voxels = torch.from_numpy(batch_voxels).permute(0, 3, 1, 2).to(self.device)
		point_coord = self.coords.clone().detach()
		point_coord = point_coord.to(self.device)
		model_out, _, _, out_m, out_b = self.bsp_network(batch_voxels, None, None, point_coord, is_training=False)	
		model_out = np.clip(np.resize(model_out.detach().cpu().numpy(),
			[self.shape_batch_size,self.sample_vox_size,self.sample_vox_size])*256, 0, 255).astype(np.uint8)
		
		image_out_size = 256
		for t in range(self.shape_batch_size):
			bsp_convex_list = []
			color_list = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]
			color_idx_list = []

			for i in range(self.gf_dim):
				min_v = np.min(model_out[t,:,:,i])
				if min_v<0.01:
					box = []
					for j in range(self.p_dim):
						if w2[j,i]>0.01:
							a = -out_m[t,0,j]
							b = -out_m[t,1,j]
							d = -out_b[t,0,j]
							box.append([a,b,d])
					if len(box)>0:
						bsp_convex_list.append(np.array(box,np.float32))
						color_idx_list.append(i%len(color_list))

			#print(bsp_convex_list)
			print(len(bsp_convex_list))
			
			#convert bspt to mesh
			vertices = []
			polygons = []
			polygons_color = []

			img_out = np.full([image_out_size,image_out_size,3],255,np.uint8)
			for i in range(len(bsp_convex_list)):
				vg, tg = digest_bsp(bsp_convex_list[i], bias=0)
				cg = color_list[color_idx_list[i]]
				for j in range(len(tg)):
					x1 = ((vg[tg[j][0]][1]+0.5)*image_out_size).astype(np.int32)
					y1 = ((vg[tg[j][0]][0]+0.5)*image_out_size).astype(np.int32)
					x2 = ((vg[tg[j][1]][1]+0.5)*image_out_size).astype(np.int32)
					y2 = ((vg[tg[j][1]][0]+0.5)*image_out_size).astype(np.int32)
					cv2.line(img_out, (x1,y1), (x2,y2), cg, thickness=1)
			
			cv2.imwrite(config.sample_dir+"/"+str(t)+"_bsp.png", img_out)

	def test_ae3(self, config):
		pass
