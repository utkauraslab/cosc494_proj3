from setuptools import find_packages, setup
from glob import glob
import os

package_name = "proj2"


setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "maps"), glob("maps/*.yaml") + glob("maps/*.pgm")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="fei",
    maintainer_email="lnnx2006@gmail.com",
    description="TODO: Package description",
    license="BSD-3-Clause",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "motion_model = proj2.motion_model:main",
            "sensor_model = proj2.sensor_model:main",
            "particle_filter = proj2.particle_filter:main",
            "resampler = proj2.resampler:main",
        ],
    },
)
