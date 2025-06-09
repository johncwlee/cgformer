learning_map = {
    'kitti': {
          0 : 0,     # "unlabeled"
          1 : 0,     # "outlier" mapped to "unlabeled" --------------------------mapped
          10: 1,     # "car"
          11: 2,     # "bicycle"
          13: 5,     # "bus" mapped to "other-vehicle" --------------------------mapped
          15: 3,     # "motorcycle"
          16: 5,     # "on-rails" mapped to "other-vehicle" ---------------------mapped
          18: 4,     # "truck"
          20: 5,     # "other-vehicle"
          30: 6,     # "person"
          31: 7,     # "bicyclist"
          32: 8,     # "motorcyclist"
          40: 9,     # "road"
          44: 10,    # "parking"
          48: 11,    # "sidewalk"
          49: 12,    # "other-ground"
          50: 13,    # "building"
          51: 14,    # "fence"
          52: 0,     # "other-structure" mapped to "unlabeled" ------------------mapped
          60: 9,     # "lane-marking" to "road" ---------------------------------mapped
          70: 15,    # "vegetation"
          71: 16,    # "trunk"
          72: 17,    # "terrain"
          80: 18,    # "pole"
          81: 19,    # "traffic-sign"
          99: 0,     # "other-object" to "unlabeled" ----------------------------mapped
          252: 1,    # "moving-car" to "car" ------------------------------------mapped
          253: 7,    # "moving-bicyclist" to "bicyclist" ------------------------mapped
          254: 6,    # "moving-person" to "person" ------------------------------mapped
          255: 8,    # "moving-motorcyclist" to "motorcyclist" ------------------mapped
          256: 5,    # "moving-on-rails" mapped to "other-vehicle" --------------mapped
          257: 5,    # "moving-bus" mapped to "other-vehicle" -------------------mapped
          258: 4,    # "moving-truck" to "truck" --------------------------------mapped
          259: 5,    # "moving-other"-vehicle to "other-vehicle" ----------------mapped
    },
    'kitti360': {
        0 : 0,    # "unlabeled"
        1 : 255,  # "ego vehicle" to ignore ------------------------------------mapped
        2 : 255,  # "rectification border" to ignore ---------------------------mapped
        3 : 255,  # "out of roi" to ignore -------------------------------------mapped
        4 : 255,  # "static" to ignore -----------------------------------------mapped
        5 : 255,  # "dynamic" to ignore ----------------------------------------mapped
        6 : 10,   # "ground" to "other-ground" ---------------------------------mapped
        7 : 7,    # "road"
        8 : 9,    # "sidewalk"
        9 : 8,    # "parking"
        10: 255,  # "rail track" to ignore -------------------------------------mapped
        11: 11,   # "building"
        12: 17,   # "wall" to "other-structure" --------------------------------mapped
        13: 12,   # "fence"
        14: 12,   # "guard rail" to "fence" ------------------------------------mapped
        15: 10,   # "bridge" to "other-ground" ---------------------------------mapped
        16: 10,   # "tunnel" to "other-ground" ---------------------------------mapped
        17: 15,   # "pole"
        18: 15,   # "polegroup" to "pole" --------------------------------------mapped
        19: 18,   # "traffic light" to "other-object" --------------------------mapped
        20: 16,   # "traffic sign"
        21: 13,   # "vegetation"
        22: 14,   # "terrain"
        23: 0,    # "sky" to "unlabeled" ---------------------------------------mapped
        24: 6,    # "person"
        25: 6,    # "rider" to "person" ----------------------------------------mapped
        26: 1,    # "car"
        27: 4,    # "truck"
        28: 5,    # "bus" to "other-vehicle" -----------------------------------mapped
        29: 1,    # "caravan" to "car" -----------------------------------------mapped
        30: 5,    # "trailer" to "other-vehicle" -------------------------------mapped
        31: 5,    # "train" to "other-vehicle" ---------------------------------mapped
        32: 3,    # "motorcycle"
        33: 2,    # "bicycle"
        34: 17,   # "garage" to "other-structure" ------------------------------mapped
        35: 17,   # "gate" to "other-structure" --------------------------------mapped
        36: 18,   # "stop" to "other-object" -----------------------------------mapped
        37: 15,   # "smallpole" to "pole" --------------------------------------mapped
        38: 18,   # "lamp" to "other-object" -----------------------------------mapped
        39: 18,   # "trash bin" to "other-object" ------------------------------mapped
        40: 18,   # "vending machine" to "other-object" ------------------------mapped
        41: 18,   # "box" to "other-object" ------------------------------------mapped
        42: 255,  # "unknown construction" to ignore ---------------------------mapped
        43: 5,    # "unknown vehicle" to "other-vehicle" -----------------------mapped
        44: 18,   # "unknown object" to "other-object" -------------------------mapped
        -1: -1,   # "license plate" to ignore ----------------------------------mapped
        255: 255,  # "ignore" to ignore -----------------------------------------mapped
    }
}