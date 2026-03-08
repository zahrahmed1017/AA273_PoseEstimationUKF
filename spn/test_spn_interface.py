import numpy as np
from spn.model import SPNWrapper, solve_pose 

# Initialize SPNWrapper
spn = SPNWrapper(
    slab_spn_root= "FinalProject/slab-spn"
    cfg_yaml= "FinalProject/slab-spn/cfg/cfg_shirt_baseline.yaml"
    checkpoint='FinalProject/slab-spn/output/efficientnet_b0.ra_in1k/baseline_incAug_20251022/model_best.pth.tar',
)