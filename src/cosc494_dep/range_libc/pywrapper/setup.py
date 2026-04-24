from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy, os, platform, sys
from os.path import join as pjoin


# Obtain the numpy include directory.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()


def check_for_flag(flag_str, truemsg=False, falsemsg=False):
    if flag_str in os.environ:
        enabled = os.environ[flag_str].lower() == "on"
    else:
        enabled = False

    if enabled and truemsg is not False:
        print(truemsg)
    elif not enabled and falsemsg is not False:
        print(falsemsg)
        print("   $ sudo " + flag_str + "=ON python setup.py install")
    return enabled


use_cuda = check_for_flag("WITH_CUDA", "Compiling with CUDA support", "Compiling without CUDA support. To enable CUDA use:")

trace = check_for_flag("TRACE", "Compiling with trace enabled for Bresenham's Line", "Compiling without trace enabled for Bresenham's Line")

print()
print("--------------")
print()

# support for compiling in clang
if platform.system().lower() == "darwin":
    os.environ["MACOSX_DEPLOYMENT_TARGET"] = platform.mac_ver()[0]
    os.environ["CC"] = "c++"


def find_in_path(name, path):
    """Find a file in a search path"""
    for directory in path.split(os.pathsep):
        binpath = pjoin(directory, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system"""
    if os.path.isdir("/usr/local/cuda-7.5"):
        home = "/usr/local/cuda-7.5"
        nvcc = pjoin(home, "bin", "nvcc")
    elif os.path.isdir("/usr/local/cuda"):
        home = "/usr/local/cuda"
        nvcc = pjoin(home, "bin", "nvcc")
    elif "CUDAHOME" in os.environ:
        home = os.environ["CUDAHOME"]
        nvcc = pjoin(home, "bin", "nvcc")
    else:
        nvcc = find_in_path("nvcc", os.environ["PATH"])
        if nvcc is None:
            raise EnvironmentError("The nvcc binary could not be located in your $PATH. " "Either add it to your path, or set $CUDAHOME")
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {"home": home, "nvcc": nvcc, "include": pjoin(home, "include"), "lib64": pjoin(home, "lib64")}

    # Python 3 fix: items() → items()
    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            raise EnvironmentError("The CUDA %s path could not be located in %s" % (k, v))

    return cudaconfig


##################### Configuration ############################

compiler_flags = ["-w", "-std=c++11", "-march=native", "-ffast-math", "-fno-math-errno", "-O3"]

nvcc_flags = ["-arch=sm_20", "--ptxas-options=-v", "-c", "--compiler-options", "'-fPIC'", "-w", "-std=c++11"]

include_dirs = ["../", numpy_include]
depends = ["../includes/*.h"]
sources = ["RangeLibc.pyx", "../vendor/lodepng/lodepng.cpp"]

CHUNK_SIZE = "262144"
NUM_THREADS = "256"

if use_cuda:
    compiler_flags.append("-DUSE_CUDA=1")
    nvcc_flags.append("-DUSE_CUDA=1")
    compiler_flags.append("-DCHUNK_SIZE=" + CHUNK_SIZE)
    nvcc_flags.append("-DCHUNK_SIZE=" + CHUNK_SIZE)
    compiler_flags.append("-DNUM_THREADS=" + NUM_THREADS)
    nvcc_flags.append("-DNUM_THREADS=" + NUM_THREADS)

    CUDA = locate_cuda()
    include_dirs.append(CUDA["include"])
    sources.append("../includes/kernels.cu")

if trace:
    compiler_flags.append("-D_MAKE_TRACE_MAP=1")


##################################################################


def customize_compiler_for_nvcc(self):
    self.src_extensions.append(".cu")

    default_compiler_so = self.compiler_so
    super_compile = self._compile

    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == ".cu":
            self.set_executable("compiler_so", CUDA["nvcc"])
            postargs = extra_postargs["nvcc"]
        else:
            postargs = extra_postargs["gcc"]

        super_compile(obj, src, ext, cc_args, postargs, pp_opts)
        self.compiler_so = default_compiler_so

    self._compile = _compile


class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)


if use_cuda:
    ext = Extension(
        "range_libc",
        sources,
        extra_compile_args={"gcc": compiler_flags, "nvcc": nvcc_flags},
        extra_link_args=["-std=c++11"],
        include_dirs=include_dirs,
        library_dirs=[CUDA["lib64"]],
        libraries=["cudart"],
        runtime_library_dirs=[CUDA["lib64"]],
        depends=depends,
        language="c++",
    )

    setup(name="range_libc", author="Corey Walsh", version="0.1", ext_modules=[ext], cmdclass={"build_ext": custom_build_ext})

else:
    setup(
        ext_modules=[
            Extension(
                "range_libc",
                sources,
                extra_compile_args=compiler_flags,
                extra_link_args=["-std=c++11"],
                include_dirs=include_dirs,
                depends=["../includes/*.h"],
                language="c++",
            )
        ],
        name="range_libc",
        author="Corey Walsh",
        version="0.1",
        cmdclass={"build_ext": build_ext},
    )
