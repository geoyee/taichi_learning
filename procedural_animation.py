import taichi as ti


ti.init(arch=ti.gpu)

n = 512  # 画布大小
canvas = ti.Vector.field(3, dtype=ti.f32, shape=(n, n))
gui = ti.GUI("Classroom homework", (n, n))

@ti.func
def frac(x):
    return x - ti.floor(x)

@ti.kernel
def paint(t: ti.f32):
    for _i, _j in canvas:
        color = ti.Vector([0.0, 0.0, 0.0])
        levels = 7
        for k in range(levels):
            block_size = 2 * 2 ** k
            # TODO:ij移动到k外面为什么不行
            i = _i + t
            j = _j + t
            p = i % block_size / block_size
            q = j % block_size / block_size
            i = i // block_size
            j = j // block_size
            brightness = (0.7 - ti.Vector([p - 0.5, q - 0.5]).norm()) * 2
            weight = 0.5 ** (levels - k - 1) * brightness
            c = frac(ti.sin(float(i * 19 + j * 98)) * 7 + t * 5e-3) * weight
            # TODO:每层的颜色有区别？
            levels_c = ti.Vector([((levels - k - 1) / 10 * c), ((k + 1) / 10 * c), (0.9 ** (k + 1) * c)])
            color += levels_c
        canvas[_i, _j] = color

t = 0.0
while True:
    t += 0.1
    paint(t)
    gui.set_image(canvas)
    gui.show()