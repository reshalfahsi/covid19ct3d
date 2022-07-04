# MIT License
# 
# Copyright (c) 2022 Resha Dwika Hefni Al-Fahsi
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================


from pathlib import Path
from setuptools import setup, find_packages


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# read version
version = {}
version_file_contents = (this_directory / "covid19ct3d" / "version.py").read_text()
exec(version_file_contents, version)


setup(
    name="covid19ct3d",
    version=version['__version__'],
    author="Resha Dwika Hefni Al-Fahsi",
    author_email='resha.alfahsi@gmail.com',
    description="COVID-19 Classification from 3D CT Images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/reshalfahsi/covid19ct3d",
    packages=find_packages(exclude=["tests"]),
    python_requires=">=3.6",
    install_requires=['numpy',
                      'ptflops',
                      'torch',
                      'opencv-python',
                      'nibabel',
                      'scipy',
                      'matplotlib',
                      'moviepy',
                      'typer'],
    include_package_data=True,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Education",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    keywords="machine-learning, deep-learning, ml, pytorch, image-segmentation, vision, medical-image-segmentation",
    entry_points={"console_scripts": ["covid19ct3d=covid19ct3d.__main__:app"]},
)
