from mjcf_urdf_simple_converter import convert
convert("inpired.xml", "/home/user/Downloads/Inspire_hand_R5.27  /Inspire_hand_R5.27/inspire_hand_r/urdf/inspire_hand_r.urdf")
# or, if you are using it in your ROS package and would like for the mesh directories to be resolved correctly, set meshfile_prefix, for example:
# convert("model.xml", "model.urdf", asset_file_prefix="package://your_package_name/model/")