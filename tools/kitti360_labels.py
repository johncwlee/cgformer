#!/usr/bin/python
#
# KITTI-360 labels
#

from collections import namedtuple


#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'kittiId'     , # An integer ID that is associated with this label for KITTI-360
                    # NOT FOR RELEASING

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'ignoreInInst', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations of instance segmentation or not

    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
    #       name                     id    kittiId,    trainId   category            catId     hasInstances   ignoreInEval   ignoreInInst   color
    Label(  'unlabeled'            ,  0 ,       -1 ,         0 , 'void'            , 0       , False        , True         , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,       -1 ,       255 , 'void'            , 0       , False        , True         , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,       -1 ,       255 , 'void'            , 0       , False        , True         , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,       -1 ,       255 , 'void'            , 0       , False        , True         , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,       -1 ,       255 , 'void'            , 0       , False        , True         , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,       -1 ,       255 , 'void'            , 0       , False        , True         , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,       -1 ,        10 , 'void'            , 0       , False        , True         , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        1 ,         7 , 'flat'            , 1       , False        , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        3 ,         9 , 'flat'            , 1       , False        , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,        2 ,         8 , 'flat'            , 1       , False        , True         , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,        10,       255 , 'flat'            , 1       , False        , True         , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        11,        11 , 'construction'    , 2       , True         , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        7 ,        17 , 'construction'    , 2       , False        , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        8 ,        12 , 'construction'    , 2       , False        , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,        30,        12 , 'construction'    , 2       , False        , True         , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,        31,        10 , 'construction'    , 2       , False        , True         , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,        32,        10 , 'construction'    , 2       , False        , True         , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        21,        15 , 'object'          , 3       , True         , False        , True         , (153,153,153) ),
    Label(  'polegroup'            , 18 ,       -1 ,        15 , 'object'          , 3       , False        , True         , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        23,        18 , 'object'          , 3       , True         , False        , True         , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        24,        16 , 'object'          , 3       , True         , False        , True         , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        5 ,        13 , 'nature'          , 4       , False        , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        4 ,        14 , 'nature'          , 4       , False        , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,        9 ,         0 , 'sky'             , 5       , False        , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,        19,         6 , 'human'           , 6       , True         , False        , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,        20,         6 , 'human'           , 6       , True         , False        , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,        13,         1 , 'vehicle'         , 7       , True         , False        , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,        14,         4 , 'vehicle'         , 7       , True         , False        , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,        34,         5 , 'vehicle'         , 7       , True         , False        , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,        16,         1 , 'vehicle'         , 7       , True         , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,        15,         5 , 'vehicle'         , 7       , True         , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,        33,         5 , 'vehicle'         , 7       , True         , False        , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,        17,         3 , 'vehicle'         , 7       , True         , False        , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,        18,         2 , 'vehicle'         , 7       , True         , False        , False        , (119, 11, 32) ),
    Label(  'garage'               , 34 ,        12,        17 , 'construction'    , 2       , True         , True         , True         , ( 64,128,128) ),
    Label(  'gate'                 , 35 ,        6 ,        17 , 'construction'    , 2       , False        , True         , True         , (190,153,153) ),
    Label(  'stop'                 , 36 ,        29,        18 , 'construction'    , 2       , True         , True         , True         , (150,120, 90) ),
    Label(  'smallpole'            , 37 ,        22,        15 , 'object'          , 3       , True         , True         , True         , (153,153,153) ),
    Label(  'lamp'                 , 38 ,        25,        18 , 'object'          , 3       , True         , True         , True         , (0,   64, 64) ),
    Label(  'trash bin'            , 39 ,        26,        18 , 'object'          , 3       , True         , True         , True         , (0,  128,192) ),
    Label(  'vending machine'      , 40 ,        27,        18 , 'object'          , 3       , True         , True         , True         , (128, 64,  0) ),
    Label(  'box'                  , 41 ,        28,        18 , 'object'          , 3       , True         , True         , True         , (64,  64,128) ),
    Label(  'unknown construction' , 42 ,        35,       255 , 'void'            , 0       , False        , True         , True         , (102,  0,  0) ),
    Label(  'unknown vehicle'      , 43 ,        36,         5 , 'void'            , 0       , False        , True         , True         , ( 51,  0, 51) ),
    Label(  'unknown object'       , 44 ,        37,        18 , 'void'            , 0       , False        , True         , True         , ( 32, 32, 32) ),
    Label(  'license plate'        , -1 ,        -1,        -1 , 'vehicle'         , 7       , False        , True         , True         , (  0,  0,142) ),
]

#--------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
#--------------------------------------------------------------------------------

# Please refer to the main method below for example usages!

# name to label object
name2label      = { label.name    : label for label in labels           }
# id to label object
id2label        = { label.id      : label for label in labels           }
# trainId to label object
trainId2label   = { label.trainId : label for label in reversed(labels) }
# KITTI-360 ID to cityscapes ID
kittiId2label   = { label.kittiId : label for label in labels           }
# category to list of label objects
category2labels = {}
for label in labels:
    category = label.category
    if category in category2labels:
        category2labels[category].append(label)
    else:
        category2labels[category] = [label]

#--------------------------------------------------------------------------------
# Assure single instance name
#--------------------------------------------------------------------------------

# returns the label name that describes a single instance (if possible)
# e.g.     input     |   output
#        ----------------------
#          car       |   car
#          cargroup  |   car
#          foo       |   None
#          foogroup  |   None
#          skygroup  |   None
def assureSingleInstanceName( name ):
    # if the name is known, it is not a group
    if name in name2label:
        return name
    # test if the name actually denotes a group
    if not name.endswith("group"):
        return None
    # remove group
    name = name[:-len("group")]
    # test if the new name exists
    if not name in name2label:
        return None
    # test if the new name denotes a label that actually has instances
    if not name2label[name].hasInstances:
        return None
    # all good then
    return name

#--------------------------------------------------------------------------------
# Main for testing
#--------------------------------------------------------------------------------

# just a dummy main
if __name__ == "__main__":
    # Print all the labels
    print("List of KITTI-360 labels:")
    print("")
    print("    {:>21} | {:>3} | {:>7} | {:>14} | {:>10} | {:>12} | {:>12}".format( 'name', 'id', 'trainId', 'category', 'categoryId', 'hasInstances', 'ignoreInEval' ))
    print("    " + ('-' * 98))
    for label in labels:
        # print("    {:>21} | {:>3} | {:>7} | {:>14} | {:>10} | {:>12} | {:>12}".format( label.name, label.id, label.trainId, label.category, label.categoryId, label.hasInstances, label.ignoreInEval ))
        print(" \"{:}\"".format(label.name))
    print("")

    print("Example usages:")

    # Map from name to label
    name = 'car'
    id   = name2label[name].id
    print("ID of label '{name}': {id}".format( name=name, id=id ))

    # Map from ID to label
    category = id2label[id].category
    print("Category of label with ID '{id}': {category}".format( id=id, category=category ))

    # Map from trainID to label
    trainId = 0
    name = trainId2label[trainId].name
    print("Name of label with trainID '{id}': {name}".format( id=trainId, name=name ))