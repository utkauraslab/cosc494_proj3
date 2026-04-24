from setuptools import setup
from glob import glob
import os

package_name = "proj1"


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
        (os.path.join("share", package_name, "plans"), glob_files("plans/*.txt")),
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
            "launch_testing",
            "launch_testing_ros",
            "pytest-timeout",
        ]
    },
    entry_points={
        "console_scripts": [
            "listener = proj1.listener:main",
            "fibonacci = proj1.fibonacci:main",
            "path_publisher = proj1.path_publisher:main",
            "pose_listener = proj1.pose_listener:main",
        ],
    },
)
