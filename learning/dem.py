import numpy as np
import taichi as ti
from osgeo import gdal


ti.init(ti.gpu)

gui = ti.GUI("dem", (1280, 640))


dem = gdal.Open("testdata/dem.tif").ReadAsArray()
dmin = np.min(dem)
dmax = np.max(dem)
dem = ((dem - dmin) / (dmax - dmin)).astype("float32")
dem_field = ti.field(ti.f32, (1280, 640))
dem_field.from_numpy(dem[:1280, :640])

color = ti.Vector([78 / 255, 203 / 255, 255 / 255])  # 海洋蓝
water_field = ti.Vector.field(3, ti.f32, (1280, 640))


@ti.kernel
def change(t: ti.f32):
    for i, j in water_field:
        if t > dem_field[i, j]:
            water_field[i, j] = 0.3 * color + 0.7 * ti.Vector([dem_field[i, j]] * 3)
        else:
            water_field[i, j] = ti.Vector([dem_field[i, j]] * 3)


t = 0.0

while True:
    change(t)
    t += 0.0005
    gui.set_image(water_field)
    gui.show()
