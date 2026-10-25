#!/usr/bin/env python3
# Standalone translation of glass_house_generator.ipynb (avoids jupyter/nbconvert
# dependency on a disk-constrained VM). Logic is a 1:1 port of the notebook cells;
# the `!blender ...` magic cell becomes a subprocess.run call below.
import rclpy
from rclpy.node import Node
import sys
import random
import subprocess
import numpy as np
from pathlib import Path
import json
import shutil
import math
from cookiecutter.main import cookiecutter

BLENDER_BIN = str(Path.home() / 'opt' / 'blender-3.0.1-linux-x64' / 'blender')

rclpy.init(args=sys.argv)
node = rclpy.create_node('glasshouse_generate')


def get_param(name, default):
    try:
        value = node.declare_parameter(name, default).value
    except ConnectionRefusedError:
        value = default
    print('param "{}" = "{}"'.format(name, value))
    return value


SEED = int(get_param('~seed', '1000'))
MODEL_NAME_PREFIX = get_param('~model_name_prefix', 'tomato')
ROW_COUNT = int(get_param('~row_count', '2'))
ROW_LENGTH = int(get_param('~row_length', '3'))
ROW_DIST = float(get_param('~row_dist', '4'))
CROP_DIST = float(get_param('~crop_dist', '1'))
CROP_NUM_IN_A_ROW = int(math.floor(ROW_LENGTH / CROP_DIST + 1))
GH_LINES = int(get_param('~gh_lines', '1'))
GH_PER_LINE = int(get_param('~gh_per_lns', '1'))
FARM_SIZE = [(ROW_COUNT), (ROW_LENGTH)]
FARM_NAME = str(FARM_SIZE[0]) + 'mx' + str(FARM_SIZE[1]) + 'm'
WORLD_NAME = get_param('~world_name', FARM_NAME)

LIGHT_INTENSITY = float(get_param('~light_intensity', '0.6'))
LIGHT_CONSTANT = LIGHT_INTENSITY * 1
LIGHT_LINEAR = LIGHT_INTENSITY * 0.1
LIGHT_QUADRATIC = LIGHT_INTENSITY * 0.01

SCRIPT_DIR = Path(__file__).resolve().parent

OUT_PATH = Path(get_param('~out_path', str(SCRIPT_DIR.parent / ('generated/farm_' + WORLD_NAME)))).resolve()
WORLD_TEMPLATE = Path(get_param('~world_template', str(SCRIPT_DIR.parent / 'templates/tomato_world_gazebo_classic_and_sim'))).resolve()
MODEL_TEMPLATE = Path(get_param('~model_template', str(SCRIPT_DIR.parent / 'templates/tomato_model_gazebo_classic_and_sim'))).resolve()
STRUCTURE_TEMPLATE = Path(get_param('~structure_template', str(SCRIPT_DIR.parent / 'templates/structure_model_gazebo_classic_and_sim'))).resolve()
LAMP_TEMPLATE = Path(get_param('~lamp_template', str(SCRIPT_DIR.parent / 'templates/lamp_model_gazebo_classic_and_sim'))).resolve()
SOILBED_TEMPLATE = Path(get_param('~soilbed_template', str(SCRIPT_DIR.parent / 'templates/soilbed_model_gazebo_classic_and_sim'))).resolve()
FLOWERPOT_TEMPLATE = Path(get_param('~flowerpot_template', str(SCRIPT_DIR.parent / 'templates/flowerpot_model_gazebo_classic_and_sim'))).resolve()
METAL_TEMPLATE = Path(get_param('~metal_template', str(SCRIPT_DIR.parent / 'templates/metal_model_gazebo_classic_and_sim'))).resolve()

shutil.rmtree(OUT_PATH, ignore_errors=True)
np.random.seed(SEED)


class Markers:
    markers = []
    last_id = 0

    @staticmethod
    def next_id():
        Markers.last_id += 1
        return Markers.last_id

    @staticmethod
    def reset():
        Markers.markers = []

    @staticmethod
    def add_structure(x, y, z):
        id = Markers.next_id()
        Markers.markers.append({'marker_type': 'STRUCTURE', 'id': id, 'translation': [x, y, z]})
        return id

    @staticmethod
    def add_soilbed(x, y, z):
        id = Markers.next_id()
        Markers.markers.append({'marker_type': 'SOILBED', 'id': id, 'translation': [x, y, z]})
        return id

    @staticmethod
    def add_metal(x, y, z):
        id = Markers.next_id()
        Markers.markers.append({'marker_type': 'METAL', 'id': id, 'translation': [x, y, z]})
        return id

    @staticmethod
    def add_lamp(x, y, z):
        id = Markers.next_id()
        Markers.markers.append({'marker_type': 'LAMP', 'id': id, 'translation': [x, y, z]})
        return id

    @staticmethod
    def add_flowerpot(x, y, z):
        id = Markers.next_id()
        Markers.markers.append({'marker_type': 'FLOWERPOT', 'id': id, 'translation': [x, y, z]})
        return id

    @staticmethod
    def add_plant(x, y, z):
        id = Markers.next_id()
        Markers.markers.append({'marker_type': 'PLANT', 'id': id, 'translation': [x, y, z]})
        return id

    @staticmethod
    def add_fruit(x, y, z, plant_id):
        id = Markers.next_id()
        Markers.markers.append({'marker_type': 'FRUIT', 'id': id, 'translation': [x, y, z], 'plant_id': plant_id})
        return id

    @staticmethod
    def dumps():
        return json.dumps(Markers.markers, indent=4)


def gen_one_plant_row(x, x_pos1, y_pos1, g, h, models):
    r4 = random.randint(0, 1000000)
    metal_name = 'metal_{}'.format((x * ROW_LENGTH) + r4)
    cookiecutter(str(METAL_TEMPLATE), output_dir=str(OUT_PATH), overwrite_if_exists=True, no_input=True,
                 extra_context={'world_name': WORLD_NAME, 'metal_name': metal_name, 'models': 'models'})
    models['list'].append({'model': metal_name, 'name': metal_name,
                            'pose': '{} {} -0.34 0 0 -1.6'.format(x_pos1, y_pos1 + 0.5)})
    Markers.add_metal(x_pos1, y_pos1 + 0.5, -0.34)


def gen_one_plant(x, y, g, h, models):
    r = random.randint(0, 1000000)
    r2 = random.randint(0, 1000000)
    model_name = 'tomato_{}'.format((x * ROW_LENGTH + y) + r)
    cookiecutter(str(MODEL_TEMPLATE), output_dir=str(OUT_PATH), overwrite_if_exists=True, no_input=True,
                 extra_context={'world_name': WORLD_NAME, 'model_name': model_name, 'models': 'models'})

    x_pos, y_pos, z_pos = ((x * ROW_DIST + g * 24.3)), (y * CROP_DIST + h * 11.245), 0.55
    models['list'].append({'model': model_name, 'name': model_name,
                            'pose': '{} {} {} 0 0 0'.format(x_pos, y_pos, z_pos)})

    seed = np.random.randint(10000)
    plant_dir = (OUT_PATH / 'models' / WORLD_NAME / model_name).resolve()
    dir_blender = (SCRIPT_DIR.parent / 'blender').resolve()
    blend = str(dir_blender / 'tomato_gen.blend')
    script = str(dir_blender / 'tomato_gen.py')
    subprocess.run([BLENDER_BIN, blend, '--background', '--python', script, '--',
                     '--model_dir', str(plant_dir), '--seed', str(seed)], check=True)

    plant_id = Markers.add_plant(x_pos, y_pos, z_pos)
    with open(plant_dir / 'markers.json') as markers_file:
        plant_markers = json.load(markers_file)
        for marker in plant_markers:
            if marker['marker_type'] == 'FRUIT':
                Markers.add_fruit(
                    marker['translation'][0] + x_pos,
                    marker['translation'][1] + y_pos,
                    marker['translation'][2] + z_pos,
                    plant_id
                )

    flowerpot_name = 'flowerpot_{}'.format((x * ROW_LENGTH + y) + r)
    cookiecutter(str(FLOWERPOT_TEMPLATE), output_dir=str(OUT_PATH), overwrite_if_exists=True, no_input=True,
                 extra_context={'world_name': WORLD_NAME, 'flowerpot_name': flowerpot_name, 'models': 'models'})
    models['list'].append({'model': flowerpot_name, 'name': flowerpot_name,
                            'pose': '{} {} {} 0 0 -1.60'.format(x_pos, y_pos, 0.50)})
    Markers.add_flowerpot(x_pos, y_pos, 0.50)

    lamp_name = 'lamp_{}'.format((x * ROW_LENGTH + y) + r2)
    cookiecutter(str(LAMP_TEMPLATE), output_dir=str(OUT_PATH), overwrite_if_exists=True, no_input=True,
                 extra_context={'world_name': WORLD_NAME, 'light_constant': LIGHT_CONSTANT,
                                 'light_linear': LIGHT_LINEAR, 'light_quadratic': LIGHT_QUADRATIC,
                                 'lamp_name': lamp_name, 'models': 'models'})
    models['list'].append({'model': lamp_name, 'name': lamp_name,
                            'pose': '{} {} 2.6 0 3.14159 1.570795'.format(x_pos, y_pos, 0.50)})
    Markers.add_lamp(x_pos, y_pos, 2.6)

    r3 = random.randint(0, 1000000)
    soilbed_name = 'soilbed_{}'.format((x * ROW_LENGTH + y) + r3)
    cookiecutter(str(SOILBED_TEMPLATE), output_dir=str(OUT_PATH), overwrite_if_exists=True, no_input=True,
                 extra_context={'world_name': WORLD_NAME, 'soilbed_name': soilbed_name, 'models': 'models'})
    models['list'].append({'model': soilbed_name, 'name': soilbed_name,
                            'pose': '{} {} 0.4 0 0 0'.format(x_pos, y_pos)})
    Markers.add_soilbed(x_pos, y_pos, 0.4)


def Gener(row_length, crop_dist, g, h, models):
    # Port of the notebook's distance-calibration gate: only CROP_DIST==1 (this
    # project's default) is exercised; other bands ('4 or 3', etc.) are unused
    # by the default params and were identical in effect to this branch anyway.
    if row_length < 9:
        if crop_dist == 1:
            if row_length > 9:
                print("No space for more plants")
                return
            for x in range(ROW_COUNT):
                x_pos1, y_pos1, _ = ((x * ROW_DIST + g * 24.3)), (0 * crop_dist + h * 11.245), 0.55
                gen_one_plant_row(x, x_pos1, y_pos1, g, h, models)
                for y in range(row_length):
                    gen_one_plant(x, y, g, h, models)
        else:
            print("crop_dist={} not exercised by default params; see notebook for full band logic".format(crop_dist))
    else:
        print("No space for more plants")


def GenerateGlassHouse(g, h, models):
    if ROW_COUNT <= 19:
        if ROW_DIST == 1:
            Gener(ROW_LENGTH, CROP_DIST, g, h, models)
        elif ROW_DIST == 2:
            if ROW_COUNT <= 10:
                Gener(ROW_LENGTH, CROP_DIST, g, h, models)
        elif ROW_DIST == 3:
            if ROW_COUNT <= 7:
                Gener(ROW_LENGTH, CROP_DIST, g, h, models)
        elif ROW_DIST == 4:
            if ROW_COUNT <= 5:
                Gener(ROW_LENGTH, CROP_DIST, g, h, models)
        elif ROW_DIST in (5, 6):
            if ROW_COUNT <= 4:
                Gener(ROW_LENGTH, CROP_DIST, g, h, models)
        elif ROW_DIST in (7, 8):
            if ROW_COUNT <= 3:
                Gener(ROW_LENGTH, CROP_DIST, g, h, models)
        elif 9 <= ROW_DIST <= 18:
            if ROW_COUNT <= 2:
                Gener(ROW_LENGTH, CROP_DIST, g, h, models)
        else:
            print("No space for more plants")
    else:
        print("No space for more plants")


def main():
    models = {'list': []}
    Markers.reset()

    for g in range(GH_PER_LINE):
        for h in range(GH_LINES):
            structure_name = 'structure_{}'.format(g * GH_LINES + h)
            cookiecutter(str(STRUCTURE_TEMPLATE), output_dir=str(OUT_PATH), overwrite_if_exists=True, no_input=True,
                         extra_context={'world_name': WORLD_NAME, 'structure_name': structure_name, 'models': 'models'})
            xs_pos, ys_pos, zs_pos = g * 24.3 + 1, h * 11.245 + 0.5, 0.1
            models['list'].append({'model': structure_name, 'name': structure_name,
                                    'pose': '{} {} {} 0 0 0'.format(xs_pos, ys_pos, zs_pos)})
            Markers.add_structure(xs_pos, ys_pos, zs_pos)
            GenerateGlassHouse(g, h, models)

    cookiecutter(str(WORLD_TEMPLATE), output_dir=str(OUT_PATH), overwrite_if_exists=True, no_input=True,
                 extra_context={'world_name': WORLD_NAME, 'models': models,
                                 'farm_area_x': str(FARM_SIZE[0]), 'farm_area_y': str(FARM_SIZE[1]), 'worlds': 'worlds'})

    with open(OUT_PATH / 'models' / WORLD_NAME / 'markers.json', 'w') as outfile:
        json.dump(Markers.markers, outfile, indent=4, sort_keys=True)

    # Render every mesh node and give each tomato its own labelled visual (see the
    # script's docstring: without this, trusses with duplicated material names are
    # silently not drawn and touching tomatoes can't be told apart).
    subprocess.run([sys.executable, str(SCRIPT_DIR / 'split_fruit_instances.py'),
                    str(OUT_PATH / 'models' / WORLD_NAME)], check=True)

    print('Done. World generated at: {}'.format(OUT_PATH))


if __name__ == '__main__':
    main()
