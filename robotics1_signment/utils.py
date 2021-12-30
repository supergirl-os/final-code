import math


def calc_anguler(pose_src, pose_dst, cur_theta):
    abs_angle = 0.
    if abs(pose_src.x - pose_dst.x) < 1E-7:
        if pose_dst.y > pose_src.y:
            abs_angle = math.pi / 2
        if pose_dst.y <= pose_dst.y:
            abs_angle = -math.pi / 2

    else:
        abs_angle = math.atan((pose_dst.y - pose_src.y) / (pose_dst.x - pose_src.x))

        if pose_dst.x < pose_src.x:
            abs_angle += math.pi

    abs_angle %= math.pi * 2
    cur_theta %= math.pi * 2

    return abs_angle - cur_theta



if __name__ == "__main__":
    class Pose:
        x = 0
        y = 0

    pose0 = Pose()
    pose1 = Pose()

    pose0.x = 0
    pose0.y = 0
    pose1.x = -1
    pose1.y = -1

    print(calc_anguler(pose0, pose1, 0))