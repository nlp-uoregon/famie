from pathlib import Path
import setuptools
from sys import platform
import os

PKG_DIR = os.path.dirname(os.path.abspath(__file__))

required = Path("requirements.txt").read_text().splitlines()

HERE = Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            package_relative_path = Path(*Path(path).parts[2:], filename)
            paths.append(str(package_relative_path))
    return paths


extra_files = []
extra_files.extend(package_files(Path('', 'src/famie', 'config')))

extra_files.extend(["api/static/bundle.js",
                    "api/templates/index.html"])

setuptools.setup(
    name="famie",
    version="0.1.0",
    author="NLP Group at the University of Oregon",
    author_email="thien@cs.uoregon.edu",
    description="FAMIE: A Fast Active Learning Framework for Multilingual Information Extraction",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/nlp-uoregon/famie",
    python_requires='>=3.6',
    install_requires=required,
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src'),
    package_data={"famie": extra_files},
    entry_points={'console_scripts': 'famie=famie.entry_points.run_app:main'},
    data_files=[('.', ["requirements.txt"])],
    license='GPL-3.0 License',
    classifiers=[
        'Development Status :: 4 - Beta',

        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Information Technology',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Text Processing',
        'Topic :: Text Processing :: Linguistic',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',

        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
