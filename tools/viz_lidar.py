import numpy as np

file_name = '/home/shounak_rtml/sensor_fusion/bevfusion/data/nuscenes/samples/LIDAR_TOP/n015-2018-07-18-11-41-49+0800__LIDAR_TOP__1531885417049716.pcd.bin'

assert file_name.endswith('.bin'), 'Unsupported filetype {}'.format(file_name)

scan = np.fromfile(file_name, dtype=np.float32)
points = scan.reshape((-1, 5))[:, :4]
print(points.shape)
print(points)
