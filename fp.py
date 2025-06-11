from numpy.ma.core import zeros
from vpython import *
import os

scene = canvas(background=color.gray(0.8), width=1200, height=500, center = vec(0,0,0), fov = 0.4)
scene.camera.pos = vec(0, 2.5, 20)

start_point = vec(-25, 3, 0)

curve(pos=[vec(-50,0,0),vec(50,0,0)], color=color.red, radius = 0.1)
arrow(pos=vec(start_point.x,0,0), axis=vec(0,start_point.y,0), shaftwidth=0.1)
# arrow(pos=vec(12, 0, 0), axis=vec(0, -1, 0), shaftwidth = 0.1)

def refraction_vector(n1, n2, v_in, normal_v):
    theta1 = acos( dot(v_in, normal_v) / (mag(v_in)*mag(normal_v)) )
    #n1 sin(theta1) = n2 sin(theta2) => theta2 = asin(n1 sin(theta1) / n2)
    theta2 =  asin(n1*sin(theta1) / n2)
    crosss = cross(v_in, normal_v)
    v_out = rotate(v_in, axis=crosss, angle=theta1-theta2)
    return v_out

def var(arr):
    N = len(arr)
    arr.sort()
    arr.pop()
    arr.pop(0)
    avg = sum(arr)/(N-2)
    V = []
    for i in arr:
        V.append((i-avg)**2)
    return sum(V)

# print(var([1,2,3,4,5]))

R = 5

# 1/f = (n-1)(1/r1+1/r2)   !! r1, r2 凸面皆為正

n = 1.5
r1, r2 = 20, 15
'''
建議正負10~30
一正一負時負的絕對值要較大，否則就穿模了！
焦距太大會不準 :(
'''

x = sqrt(r1 ** 2 - R ** 2)
y = sqrt(r2 ** 2 - R ** 2)
c1 = vec(x*r1/abs(r1), 0, 0)
c2 = vec(-y*r2/abs(r2), 0, 0)

big_ball = sphere(pos=vec(0, 0, 0), radius=50, color=color.green, opacity=0.1)

if r1<0 and r2<0:
    d = 0.001
    c1 = vec(r1-d, 0, 0)
    c2 = vec(-r2+d, 0 ,0)
    cy = cylinder(pos=vec(0, -abs(r1+r2), 0), color=color.green, opacity=0.1, axis=vec(0, abs(r1+r2)*2, 0), radius=abs(r1+r2)/4)
    # big_ball.visible = False


b1 = sphere(pos=c1, radius=abs(r1), color=color.yellow, opacity=0.1*r1/abs(r1), shininess=0.8)
b2 = sphere(pos=c2, radius=abs(r2), color=color.yellow, opacity=0.1*r2/abs(r2), shininess=0.8)

slope = zeros(9)
y = zeros(9)
turn = zeros(9, dtype=vector)

X = 3*abs(r1)

for angle in range(-7, 2):
    i = angle+7
    ray = sphere(pos=start_point, color=color.blue, radius=0.01, make_trail=True)
    ray.v = vector(cos(angle / 40.0), sin(angle / 40.0), 0)
    dt = 0.002

    left = False
    right = False
    MIN = 1e9

    while ray.pos.x < 50:
        rate(2000)
        ray.pos += ray.v * dt

        if r1<0 and mag(ray.pos - c1) >= abs(r1) and ray.pos.x >= c1.x and not left:
            left = True
            ray.v = refraction_vector(1, n, ray.v, ray.pos - c1)
            # print("rf1", ray.pos.x)
        if r1>0 and mag(ray.pos - c1) <= r1 and not left:
            left = True
            ray.v = refraction_vector(1, n, ray.v, c1 - ray.pos)
            # print("rf2", ray.pos.x)

        if r2<0 and mag(ray.pos - c2) <= abs(r2) and not right:
            right = True
            ray.v = refraction_vector(n, 1, ray.v, c2 - ray.pos)
            turn[i] = vec(ray.pos.x, ray.pos.y, 0)
            # print("rf3", ray.pos.x)
        if r2>0 and mag(ray.pos - c2) >= r2 and ray.pos.x >= c2.x and not right:
            right = True
            ray.v = refraction_vector(n, 1, ray.v, ray.pos - c2)
            turn[i] = vec(ray.pos.x, ray.pos.y, 0)
            # print("rf4", ray.pos.x)

        # print(ray.v)

        if abs(ray.pos.x - X) < MIN:
            MIN = abs(ray.pos.x - X)
            slope[i] = ray.v.y/ray.v.x
            y[i] = ray.pos.y

f_theory = 1 / ((n - 1) * (1 / r1 + 1 / r2))
print("f焦距 理論值: ", f_theory)

# print(slope)
# print(y)

l, r = -1000, 1000

# 找最接近交點的值
for _ in range(10000):

    ml, mr = l + (r - l) / 3, l + (r - l) * 2 / 3
    a, b = [], []
    for i in range(0, 9):
        a.append(y[i] + slope[i]*(ml-X)) # x = ml時的值
        b.append(y[i] + slope[i]*(mr-X)) # x = mr時的值
    # print(var(a), var(b), l, r)
    if var(a)>var(b):
        l = ml
    else:
        r = mr

q_calculated = (l + r) / 2

print("f焦距 計算值: ", 1 / (1 / abs(start_point.x) + 1 / q_calculated))
print("q像距 理論值： ", 1 / (1 / f_theory - 1 / abs(start_point.x)))
print("q像距 計算值： ", q_calculated)

Y = []
for i in range(0, 9):
    Y.append(y[i] + slope[i]*(q_calculated-X)) # x = q時的值
Y.sort()
Y.pop()
Y.pop()

arrow(pos=vec(q_calculated,0,0), axis=vec(0,sum(Y)/7,0), shaftwidth=0.3, color=color.yellow)


if q_calculated < 0:
    for i in range(0, 9):
        ray = sphere(pos=vec(turn[i].x, turn[i].y, 0), color=color.orange, radius=0.01, make_trail=True)
        ray.v = -vector(1, slope[i], 0)*10
        dt = 0.002

        while ray.pos.x > -50:
            rate(3000)
            ray.pos += ray.v * dt

os.system("pause")




