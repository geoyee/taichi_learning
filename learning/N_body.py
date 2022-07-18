import numpy as np
from math import pi as PI
import taichi as ti


ti.init(ti.gpu)

gui = ti.GUI("N-body problem", (640, 640))


G = 9.8  # 引力常数
N = 500  # 星球数量
galaxy = [0.5, 0.3, 0.4]  # 星系大小（缩放，偏移，大小）
init_vel = 120  # 初始速度
color = np.array([0xFFFFFF, 0xFFFFCC, 0xFFFF99, 0xFFFF66, 0xFFFF33, 
                  0xFFFF00, 0xFFCCFF, 0xFFCCCC, 0xFFCC99, 0xFFCC66, 
                  0xFFCC33, 0xFFCC00, 0xFF99FF, 0xFF99CC, 0xFF9999, 
                  0xFF9966, 0xFF9933, 0xFF9900, 0xFF66FF, 0xFF66CC], dtype=np.uint32)

h = 1e-5  # 时间步
substepping = 10

m = ti.field(ti.i32, N)  # 质量
pos = ti.Vector.field(2, ti.f32, N)  # 位置
vel = ti.Vector.field(2, ti.f32, N)  # 速度
force = ti.Vector.field(2, ti.f32, N)  # 受力


@ti.kernel
def initialize():
    center = ti.Vector([0.5, 0.5])  # 区域中心为中点
    for i in range(N):
        theta = ti.random() * 2 * PI
        r = (ti.sqrt(ti.random()) * galaxy[0] + galaxy[1]) * galaxy[2]
        offset = r * ti.Vector([ti.cos(theta), ti.sin(theta)])
        pos[i] = center + offset  # 计算初始位置
        # 计算初始速度（带方向）
        vel[i] = [-offset.y, offset.x]
        vel[i] *= init_vel
        m[i] = ti.cast((ti.random() * 19 + 1), ti.i32)  # 计算质量

@ti.kernel
def compute_force():
    for i in range(N):
        force[i] = ti.Vector([0.0, 0.0])  # 清除力
    # 重新计算万有引力
    for i in range(N):
        _compute_force(i)

@ti.func
def _compute_force(i: ti.i32):
    # 利用外层循环加速
    for j in range(N):
        if i != j:
            diff = pos[i] - pos[j]
            r = diff.norm(1e-5)
            f = -G * (m[i] * m[j]) * (1.0 / r) ** 3 * diff
            force[i] += f

@ti.kernel
def update():
    dt = h / substepping
    for i in range(N):
        vel[i] += (dt * force[i] / m[i])  # 更新速度
        pos[i] += (dt * vel[i])  # 更新位置


initialize()

while gui.running:
    for i in range(substepping):
        compute_force()
        update()
    gui.clear(0x112F41)
    gui.circles(pos.to_numpy(), color=color[m.to_numpy()], radius=m.to_numpy())  # 质量大半径大
    gui.show()
