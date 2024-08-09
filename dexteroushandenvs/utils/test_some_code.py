# pyvirtualdisplay/examples/screenshot.py

"Create screenshot of xmessage in background using 'smartdisplay' submodule"
from easyprocess import EasyProcess

from pyvirtualdisplay.smartdisplay import SmartDisplay

# 'SmartDisplay' instead of 'Display'
# It has 'waitgrab()' method.
# It has more dependencies than Display.
# with SmartDisplay(use_xauth=True, extra_args=['-screen']) as disp:
with SmartDisplay(manage_global_env=False) as disp:
    img = disp.grab()
    print(img)
img.save("xmessage.png")