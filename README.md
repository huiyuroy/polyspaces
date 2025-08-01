# Polyspaces Dataset

## Overview
This dataset is designed for redirected walking simulation tests. It contains scene descriptions and walking trajectories for evaluating navigation algorithms in virtual environments. The data is particularly useful for research in virtual reality, robotics, and spatial computing.

## Data Organization
- All files are in JSON format
- Data is organized into two main categories:
  1. **Scene files**: Describe virtual environments with boundaries and obstacles
     - Located in `vir/`, `phy/`, and `rl/` directories
  2. **Trajectory files**: Contain walking paths through the scenes
     - Located in the main dataset directory

## File Descriptions

### 1. Scene File
Describes a virtual environment with boundaries and obstacles.

**Structure:**
```json
{
  "name": "scene_name",
  "bounds": [
    {
      "is_out_bound": boolean,
      "points": [[x1,y1], [x2,y2], ...],
      "center": [x,y],
      "barycenter": [x,y],
      "cir_rect": [[p1], [p2], [p3], [p4]]
    }
  ],
  "max_size": [width, height],
  "out_bound_conv": {
    "vertices": [[x1,y1], [x2,y2], ...],
    "center": [x,y],
    "barycenter": [x,y],
    "cir_circle": [[center_x, center_y], radius],
    "in_circle": [[center_x, center_y], radius]
  },
  "out_conv_hull": { /* convex hull properties */ },
  "scene_center": [x,y]
}
```

**Key Components:**
- `bounds`: Defines environment boundaries and obstacles
  - `is_out_bound`: True for outer boundaries, False for obstacles
  - `points`: Polygon vertices defining the shape
  - `cir_rect`: Bounding rectangle coordinates
- `max_size`: Maximum dimensions of the scene
- Geometric properties: Centers, barycenters, convex hulls, and bounding circles

### 2. Trajectory File
Contains walking paths through virtual environments.

**Structure:**
```json
{
  "id": trajectory_id,
  "type": "trajectory_type",
  "targets": [
    [x1,y1],
    [x2,y2],
    ...
  ]
}
```

**Key Components:**
- `id`: Unique trajectory identifier
- `type`: Trajectory generation method (e.g., "absolute random")
- `targets`: Sequence of 2D coordinates representing the path

## Data Access
1. **Clone the repository**:
   ```bash
   git clone https://github.com/huiyuroy/polyspaces.git
   ```

2. **Download virtual walking trajectories**:
   - [Download Link](https://pan.baidu.com/s/1-smQgwwctm__21C8IyzOBA?pwd=7qhd)
   - Password: `7qhd`

3. **Merge datasets**:
   - Downloading and unzipping the trajectory files.
   - Merge the trajectory data with the scene files in the polyspaces directory.


## Dataset Structure
```
polyspaces/
├── vir/                   # Virtual scene files
│   ├── test1.json
│   ├── .....
│   └── simu_trajs 
│         ├── abs_rand_0.json        # Trajectory files
│         └── ... 
├── phy/                   # Physical scene files
├── rl/                   
└── ... 
```

## License
This dataset is available for research purposes. 

For questions or additional information, please contact: [huiyuroy@163.com]