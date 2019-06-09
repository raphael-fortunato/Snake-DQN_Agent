from math import sqrt, acos, radians, cos, degrees

def CalcDistance(a1,a2, b1, b2):
    x_dist = (a1 - a2)**2
    y_dist = (b1 -b2)**2
    return sqrt(x_dist+ y_dist)


def CalcAngle(dir, distC):
    if distC ==0:
            return 0
    if dir[1] != 0:
        distB = CalcDistance(9,9,9,20) 
    elif dir[0] != 0:
        distB = CalcDistance(9,9,9,20)
    angle = acos(distB/ distC)
    radian = degrees(angle)
    print(f"angle_distance: { ((90 - radian)/90) * distC}, DistC: {distC}, DistB: {distB}, radian: {radian}")



dir0 = (-1,0)
distanceC = CalcDistance(9,15,9,0)
CalcAngle(dir0, distanceC)