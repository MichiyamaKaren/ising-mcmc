# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt

from isingmc import IsingModel2D

# %%
sl = 3
model = IsingModel2D(sl)
# %%
edge_length = 1000
canvas = np.ones((sl*edge_length, (sl+1)*edge_length, 3), dtype='uint8') * 255

shift_scale = 100
for i, site in enumerate(model.vertexes):
    ii, jj = model.i_to_ij(i)
    y = int((0.5+ii)*edge_length) + np.random.randint(-shift_scale, shift_scale)
    x = int((0.5+jj)*edge_length) + np.random.randint(-shift_scale, shift_scale)
    site.coord = (y, x)
    site_radius = edge_length//20
    cv2.circle(canvas, site.coord, radius=site_radius, color=(0, 0, 255), thickness=-40)

    text_scale = 5
    cv2.putText(
        canvas, f'S={site.spin:d}', (x+site_radius, y-site_radius),
        fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=text_scale, color=(0,0,255), thickness=10)
    (width, height), _ = cv2.getTextSize('S=', cv2.FONT_HERSHEY_COMPLEX, fontScale=text_scale, thickness=10)
    cv2.putText(
        canvas, f'E={site.site_energy():d}', (x+site_radius, y-site_radius-height),
        fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=text_scale, color=(0,0,255), thickness=10)

for edge in model.edges:
    cv2.line(canvas, edge.headvex.coord, edge.tailvex.coord,
             color=(255, 0, 0), thickness=20)

plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
plt.axis(False)
# %%
model.add_edge(0, 0, 2, 2)
model.add_edge(2, 2, 3, 4)
# %%
