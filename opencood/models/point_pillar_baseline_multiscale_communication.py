# -*- coding: utf-8 -*-
# Author: Quanmin Wei, Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib
# Support F-Cooper, Self-Att, DiscoNet(wo KD), V2VNet, V2XViT, When2comm

import torch.nn as nn
from icecream import ic
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone 
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone 
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.fusion_in_one import MaxFusion, AttFusion, DiscoFusion, V2VNetFusion, V2XViTFusion, When2commFusion
from opencood.utils.transformation_utils import normalize_pairwise_tfm

class PointPillarBaselineMultiscaleCommunication(nn.Module):
    def __init__(self, args):
        super(PointPillarBaselineMultiscaleCommunication, self).__init__()

        ## Initialize communication network based on the specified mode
        communication_mode = args.get('communication_mode', None)
        if communication_mode is None:
            raise ValueError("communication_mode must be specified in the config file.")
        
        if communication_mode == 'ermvp':
            ic("Using ERMVP communication mode.")
            from opencood.extensions.ermvp_communication import ERMVPCommunication 
            self.communication_network = ERMVPCommunication(channels=64, topk_ratio=0.1, cluster_sample_ratio=0.2)
        elif communication_mode == 'where2comm':
            ic("Using Where2Comm communication mode.")
            from opencood.extensions.where2comm_communication import W2CCommunication
            self.communication_network = W2CCommunication(topk_ratio=0.1, use_smooth=True, use_ste=False, out_channels=256)
            print("topk_ratio=01")
        elif communication_mode == 'naivecomm':
            ic("Using Naive Communication mode.")
            from opencood.extensions.naive_communication import NaiveCommunication
            self.communication_network = NaiveCommunication(channels=64, target_channels=4)
        else:
            raise ValueError(f"Unsupported communication mode: {communication_mode}. Supported modes are 'ermvp', 'where2comm', and 'naivecomm'.")
        
        ic(self.communication_network)

        self.communication_mode = communication_mode
        ## End of communication network initialization

        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        is_resnet = args['base_bev_backbone'].get("resnet", True) # default true
        if is_resnet:
            self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64) # or you can use ResNetBEVBackbone, which is stronger
        else:
            self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64) # or you can use ResNetBEVBackbone, which is stronger
        self.voxel_size = args['voxel_size']

        self.fusion_net = nn.ModuleList()
        for i in range(len(args['base_bev_backbone']['layer_nums'])):
            if args['fusion_method'] == "max":
                self.fusion_net.append(MaxFusion())
            if args['fusion_method'] == "att":
                self.fusion_net.append(AttFusion(args['att']['feat_dim'][i]))
        self.out_channel = sum(args['base_bev_backbone']['num_upsample_filter'])

        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.out_channel = args['shrink_header']['dim'][-1]

        self.compression = False
        if "compression" in args:
            self.compression = True
            self.naive_compressor = NaiveCompressor(64, args['compression'])

        self.cls_head = nn.Conv2d(self.out_channel, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(self.out_channel, 7 * args['anchor_number'],
                                  kernel_size=1)
        self.use_dir = False
        if 'dir_args' in args.keys():
            self.use_dir = True
            self.dir_head = nn.Conv2d(self.out_channel, args['dir_args']['num_bins'] * args['anchor_number'], kernel_size=1) # BIN_NUM = 2
 

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        # calculate pairwise affine transformation matrix
        _, _, H0, W0 = batch_dict['spatial_features'].shape # original feature map shape H0, W0
        normalized_affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], H0, W0, self.voxel_size[0])

        spatial_features = batch_dict['spatial_features']

        if self.compression: # torch.Size([2, 64, 200, 704])
            spatial_features = self.naive_compressor(spatial_features)

        ## Communication Network
        if self.communication_mode == 'ermvp':
            # Use the ERMVP communication network
            spatial_features = self.communication_network.communication(x=spatial_features, record_len=record_len, cls_head=self.cls_head)
        elif self.communication_mode == 'where2comm':
            # Use the Where2Comm communication network
            spatial_features = self.communication_network(spatial_features, record_len, self.cls_head)
        elif self.communication_mode == 'naivecomm':
            # Use the Naive Communication Network
            spatial_features = self.communication_network(spatial_features, record_len)
        ### end

        # multiscale fusion
        feature_list = self.backbone.get_multiscale_feature(spatial_features)
        fused_feature_list = []
        for i, fuse_module in enumerate(self.fusion_net):
            fused_feature_list.append(fuse_module(feature_list[i], record_len, normalized_affine_matrix))
        fused_feature = self.backbone.decode_multiscale_feature(fused_feature_list) 

        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)

        psm = self.cls_head(fused_feature) # torch.Size([1, 256, 100, 352])->torch.Size([1, 2, 100, 352])
        rm = self.reg_head(fused_feature)

        output_dict = {'cls_preds': psm,
                       'reg_preds': rm,
                       'KL_loss': None,
                       'RL_loss': None
                       }

        if self.use_dir:
            output_dict.update({'dir_preds': self.dir_head(fused_feature)})

        return output_dict
