OCTO_INPUT_WIDTH = 256
OCTO_INPUT_HEIGHT = 256


# task names from mani_skills
REGISTERED_TASKS = (
    "TriFingerRotateCubeLevel0-v1",
    "TriFingerRotateCubeLevel1-v1",
    "TriFingerRotateCubeLevel2-v1",
    "TriFingerRotateCubeLevel3-v1",
    "TriFingerRotateCubeLevel4-v1",
    "OpenCabinetDoor-v1",
    "OpenCabinetDrawer-v1",
    "RotateValveLevel0-v1",
    "RotateValveLevel1-v1",
    "RotateValveLevel2-v1",
    "RotateValveLevel3-v1",
    "RotateValveLevel4-v1",
    "RotateSingleObjectInHandLevel0-v1",
    "RotateSingleObjectInHandLevel1-v1",
    "RotateSingleObjectInHandLevel2-v1",
    "RotateSingleObjectInHandLevel3-v1",
    "StackCube-v1",
    "TurnFaucet-v1",  # ARTNET_MOBILITY["model_urdf_paths"][id] KeyError: '5001'
    "PickCube-v1",
    "PickSingleYCB-v1",
    "PushCube-v1",
    "RollBall-v1",
    "PushT-v1",
    "PokeCube-v1",
    "PlugCharger-v1",
)

import torch.nn as nn

ACTIVATION_LAYER_MAPPING = {
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "ELU": nn.ELU,
    "SELU": nn.SELU,
    "PReLU": nn.PReLU,
    "RReLU": nn.RReLU,
    "CELU": nn.CELU,
    "GLU": nn.GLU,
    "SiLU": nn.SiLU,
    "GELU": nn.GELU,
}
