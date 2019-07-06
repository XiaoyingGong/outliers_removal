import numpy as np

AB = np.array([1,-3,5,-1])
CD = np.array([4,1,4.5,4.5])
EF = np.array([2,5,-2,6])
PQ = np.array([-3,-4,1,-6])

MN = np.array([194.81637573, 113.3913269,  201.05604553,  112.81473541])
DF = np.array([194.81637573, 113.3913269,   225.74801636,  154.33976746])

def angle(v1, v2):
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]
    angle1 = np.arctan2(dy1, dx1)
    angle1 = angle1 * 180/np.pi
    # print(angle1)
    angle2 = np.arctan2(dy2, dx2)
    angle2 = angle2 * 180/np.pi
    # print(angle2)
    if angle1*angle2 >= 0:
        included_angle = abs(angle1-angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    return included_angle

ang1 = angle(AB, CD)
print("AB和CD的夹角")
print(ang1)
ang2 = angle(AB, EF)
print("AB和EF的夹角")
print(ang2)
ang3 = angle(AB, PQ)
print("AB和PQ的夹角")
print(ang3)
ang4 = angle(CD, EF)
print("CD和EF的夹角")
print(ang4)
ang5 = angle(CD, PQ)
print("CD和PQ的夹角")
print(ang5)
ang6 = angle(EF, PQ)
print("EF和PQ的夹角")
print(ang6)
ang7 = angle(MN, DF)
print("MN和DF的夹角")
print(ang7)
