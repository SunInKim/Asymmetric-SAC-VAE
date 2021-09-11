from collections import namedtuple
from attrdict import AttrDict
import functools

def setup_panda(p, robotID):
    controlJoints = ["panda_joint1","panda_joint2",
                     "panda_joint3", "panda_joint4",
                     "panda_joint5", "panda_joint6", "panda_joint7"]
                      # "robotiq_85_left_knuckle_joint",
                     # "robotiq_85_right_knuckle_joint",
                     #        "robotiq_85_left_inner_knuckle_joint",
                     #        "robotiq_85_right_inner_knuckle_joint",
                     #        "robotiq_85_left_finger_tip_joint",
                     #        "robotiq_85_right_finger_tip_joint",
                     #        "end_effector_fixed_joint"]
    jointTypeList = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
    numJoints = p.getNumJoints(robotID)
    print(numJoints)
    jointInfo = namedtuple("jointInfo", 
                           ["id","name","type","lowerLimit","upperLimit","maxForce","maxVelocity","controllable"])
    joints = AttrDict()
    for i in range(numJoints):
        info = p.getJointInfo(robotID, i)
        jointID = info[0]
        jointName = info[1].decode("utf-8")
        jointType = jointTypeList[info[2]]
        jointLowerLimit = info[8]
        jointUpperLimit = info[9]
        jointMaxForce = info[10]
        jointMaxVelocity = info[11]
        controllable = True if jointName in controlJoints else False
        info = jointInfo(jointID,jointName,jointType,jointLowerLimit,
                         jointUpperLimit,jointMaxForce,jointMaxVelocity,controllable)
        if info.type=="REVOLUTE": # set revolute joint to static
            p.setJointMotorControl2(robotID, info.id, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
        joints[info.name] = info
        
    return joints, controlJoints

