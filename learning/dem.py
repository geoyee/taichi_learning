import numpy as np
import taichi as ti
from osgeo import gdal

ti.init(ti.gpu)

gui = ti.GUI("dem", (640, 640))


dem = gdal.Open("testdata/UH17_GEM051.tif").ReadAsArray()
dmin = np.min(dem)
dem[dem > 1e38] = dmin  # 去除无效值
dmax = np.max(dem)
dem = ((dem - dmin) / (dmax - dmin)).astype("float32")
dem_field = ti.field(ti.f32, (640, 640))


dem_field.from_numpy(dem[:640, :640])


while True:
    gui.set_image(dem_field)
    gui.show()
