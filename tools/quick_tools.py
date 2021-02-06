import numpy as np


def process_ligand_poses(ligand_poses):

    new_ligand_pose = []
    for i, pose in enumerate(ligand_poses):

        if pose.startswith('r'):
            pose = pose[1:]

        idx = pose.index('_')
        pose = pose[:idx]

        new_ligand_pose.append(int(pose))

    return new_ligand_pose

