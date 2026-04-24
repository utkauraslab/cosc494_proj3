from setuptools import setup
from glob import glob
import os

package_name = "mushr_sim"


def glob_files(pattern):
  return [p for p in glob(pattern, recursive=True) if os.path.isfile(p)]


setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob_files("launch/*.py")),
        (os.path.join("share", package_name, "maps"), glob_files("maps/**/*")),
        (os.path.join("share", package_name, "config"), glob_files("config/**/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="fei",
    maintainer_email="fliu33@utk.edu",
    description="ROS2 simulation package for MuSHR",
    license="BSD-3-Clause",
    entry_points={
        "console_scripts": [
            "mushr_sim = mushr_sim.mushr_sim:main",
            "fake_vesc_driver = mushr_sim.fake_vesc_driver:main",
            "keyboard_teleop_terminal = mushr_sim.keyboard_teleop_terminal:main",
            "clicked_point_to_reposition = mushr_sim.clicked_point_to_reposition:main",
            "fake_localization = mushr_sim.fake_localization:main",
        ],
    },
)
