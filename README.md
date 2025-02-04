# ContextualFusion: Context-Based Multi-Sensor Fusion for 3D Object Detection in Adverse Operating Conditions

This repository contains the code and resources for the paper:

**"ContextualFusion: Context-Based Multi-Sensor Fusion for 3D Object Detection in Adverse Operating Conditions"**  
*by Shounak Sural, Nishad Sahu, Ragunathan Rajkumar*  
Published at **IEEE Intelligent Vehicles Symposium (IV) 2024**, South Korea  
[[Read the paper](https://arxiv.org/abs/2404.14780)]

---

### Overview

This project is based on the repository from [MIT-Han Lab's BEVFusion](https://github.com/mit-han-lab/bevfusion) and has been extended to develop the ContextualFusion framework.

### AdverseOp3D Dataset

Access the **AdverseOp3D dataset**:  
[Download here](https://cmu.box.com/s/xq4tkgefljakzo75drydvj8eiie8cgju)

### Pretrained Models

Pretrained models are available for download:  
[Download models](https://cmu.box.com/s/au4v02xcy1iwjv9hsrn70ej6ddkh40w1)

- For night-time evaluation, use the model: `CF_Night_trained_NuScenes.pth`

### Evaluation Command

To run the evaluation on the NuScenes dataset at night-time, use the following command:

```bash
torchpack dist-run -np 2 python tools/test.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml models/CF_Night_trained_NuScenes.pth --eval bbox
```

### Citation

If you find this project useful, please cite the paper in the following format-
```
@INPROCEEDINGS{10588584,
  author={Sural, Shounak and Sahu, Nishad and Rajkumar, Ragunathan Raj},
  booktitle={2024 IEEE Intelligent Vehicles Symposium (IV)}, 
  title={ContextualFusion: Context-Based Multi-Sensor Fusion for 3D Object Detection in Adverse Operating Conditions}, 
  year={2024},
  volume={},
  number={},
  pages={1534-1541},
  keywords={Solid modeling;Three-dimensional displays;Laser radar;Lighting;Object detection;Logic gates;Cameras;Autonomous Vehicles;3D Object Detection;Night-time Perception;Adverse Weather;Contextual Fusion},
  doi={10.1109/IV55156.2024.10588584}
}
```
