# Copyright 2024 by Yaopu Zhao, Beihang University, School of Automation Science and Electrical Engineering.
# All rights reserved.
# This file is part of the edge-nicp,
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.

import torch
import io3d
import json
import numpy as np
from bfm import MorphableModel
from utils import batch_vertex_sample
from edge_nicp import tranformAtoB, non_rigid_icp_edge


device = torch.device('cuda:0')
fine_config = json.load(open('config/fine_grain.json'))


model = MorphableModel(device=device)



landmark_id_84 = [21,25,35,31, 83,40,41,43,4,3,1,0,7,5,8,9,11,12,13,15,60,49,50,51,52,53,64,55,57,59,61,62,63,65,66,67]
landmark_id_68 = [17,21,22,26, 30,31,33,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,57,59,61,62,63,65,66,67]


meshes = io3d.load_obj_as_mesh('testdata/astface_show/show_au27.obj', device = device)
target_lm_np = np.loadtxt('testdata/astface_show/show_au27_landmarks.txt').astype(np.float32)[landmark_id_84]
target_lm = torch.unsqueeze(torch.from_numpy(target_lm_np), 0).to(device)


with torch.no_grad():
    bfm_path = 'result/flame_to_bfm.obj'
    bfm_meshes = io3d.load_obj_as_mesh(bfm_path, device = device)
    bfm_lm_index = torch.unsqueeze(model.kpt_inds[landmark_id_68], 0).to(device) 
    bfm_lm = torch.tensor(batch_vertex_sample(bfm_lm_index, bfm_meshes.verts_padded()), dtype=torch.float32).to(device)
    numpy_bfm_lm = bfm_lm.detach().cpu().numpy()[0]

    after_target, after_target_lm = tranformAtoB(bfm_meshes, meshes, bfm_lm, target_lm, device)

registered_mesh = non_rigid_icp_edge(bfm_meshes, after_target, bfm_lm_index, after_target_lm,fine_config, device, with_edge=True)
io3d.save_meshes_as_objs(['result/out_mesh.obj'], registered_mesh, save_textures = False)