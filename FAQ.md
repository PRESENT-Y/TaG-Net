## Environment
```bash
conda install pytorch==0.4.1 torchvision cuda90 -c pytorch
```
```bash
conda install -c intel mkl_fft==1.0.15
```
```bash
pip install -i https://pypi.doubanio.com/simple/ -r requirements.txt
```


## Mayavi 
- ``ERROR: Failed building wheel for mayavi. ModuleNotFoundError: No module named 'vtk'.''

```bash
pip install vtk
```

- ``Could not import backend for traitsui. Make sure you have a suitable UI toolkit like PyQt/PySide or wxPython installed.''

```bash
pip install pyqt5
```
- ``qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found. Available platform plugins are: eglfs, linuxfb, minimal, minimalegl, offscreen, vnc, wayland-egl, wayland, wayland-xcomposite-egl, wayland-xcomposite-glx, webgl, xcb.''

```bash
sudo apt install libxcb-xinerama0 
```
