#!/usr/bin/env python
# coding: utf-8

import json

num_of_gd_layers = [3, 4, 5]
dim_of_gd_layers = [300, 400, 500]
z_dims = [8, 12, 16]
config_path = '/home/jakmic/Projekty/dose3d-phsp/GAN/CGAN/config_001.json'
new_config_path = '/home/jakmic/Projekty/dose3d-phsp/GAN/CGAN/config_001_new.json'


def load_current_config_state(config_path):
    with open(config_path) as json_file:
        config_json_object = json.load(json_file)
        current_num_of_d_layer = config_json_object['d_layers']
        current_d_dim = config_json_object['d_dim']
        current_z_dim = config_json_object['z_dim']
    return current_num_of_d_layer, current_d_dim, current_z_dim


class MyGridSearch:
    def __init__(self, num_of_gd_layers, dim_of_gd_layers, z_dims, current_num_of_gd_layer, current_dim_of_gd_layer, current_z_dim):
        self.num_of_gd_layers = num_of_gd_layers
        self.dim_of_gd_layers = dim_of_gd_layers
        self.z_dims = z_dims
        self.stop = False

        self.current_num_of_gd_layer = current_num_of_gd_layer
        self.current_dim_of_gd_layer = current_dim_of_gd_layer
        self.current_z_dim = current_z_dim

    def __iter__(self):
        # print("TO ja")
        return self

    def __next__(self):
        for num_of_gd_layer in self.num_of_gd_layers:
            for dim_of_gd in self.dim_of_gd_layers:
                for z_dim in self.z_dims:
                    if self.stop == True:
                        self.stop = False
                        self.current_num_of_gd_layer = num_of_gd_layer
                        self.current_dim_of_gd_layer = dim_of_gd
                        self.current_z_dim = z_dim
                        return num_of_gd_layer, dim_of_gd, z_dim
                    if self.current_num_of_gd_layer == num_of_gd_layer and self.current_dim_of_gd_layer == dim_of_gd and self.current_z_dim == z_dim:
                        self.stop = True
        raise StopIteration


def create_config_with_new_state(new_config_path, new_config_state_json_object):
    with open(new_config_path, 'w') as outfile:
        json.dump(new_config_state_json_object, outfile, indent=4)
    print("New config file created")


current_num_of_d_layer, current_d_dim, current_z_dim = load_current_config_state(
    config_path=config_path)
iter_mygridsearch = MyGridSearch(num_of_gd_layers=num_of_gd_layers, dim_of_gd_layers=dim_of_gd_layers, z_dims=z_dims,
                                 current_num_of_gd_layer=current_num_of_d_layer, current_dim_of_gd_layer=current_d_dim, current_z_dim=current_z_dim)
new_num_of_d_layer, new_d_dim, new_z_dim = next(iter_mygridsearch)
with open(config_path) as json_file:
    config_json_object = json.load(json_file)
new_state_config_json_object = config_json_object
new_state_config_json_object['d_layers'] = new_num_of_d_layer
new_state_config_json_object['g_layers'] = new_num_of_d_layer
new_state_config_json_object['d_dim'] = new_d_dim
new_state_config_json_object['g_dim'] = new_d_dim
new_state_config_json_object['z_dim'] = new_z_dim
create_config_with_new_state(new_config_path=new_config_path,
                             new_config_state_json_object=new_state_config_json_object)
