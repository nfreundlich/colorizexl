import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

REQUIRED = [
    "absl-py==0.9.0",
    "appdirs==1.4.3",
    "appnope==0.1.0",
    "astor==0.8.1",
    "attrs==19.3.0",
    "backcall==0.1.0",
    "black==19.10b0",
    "bleach==3.1.4",
    "cachetools==4.1.0",
    "certifi==2020.4.5.1",
    "chardet==3.0.4",
    "click==7.1.1",
    "cycler==0.10.0",
    "decorator==4.4.2",
    "defusedxml==0.6.0",
    "entrypoints==0.3",
    "gast==0.2.2",
    "google-auth==1.13.1",
    "google-auth-oauthlib==0.4.1",
    "google-pasta==0.2.0",
    "grpcio==1.28.1",
    "h5py==2.10.0",
    "idna==2.9",
    "imageio==2.8.0",
    "importlib-metadata==1.6.0",
    "ipykernel==5.2.0",
    "ipython==7.13.0",
    "ipython-genutils==0.2.0",
    "ipywidgets==7.5.1",
    "jedi==0.16.0",
    "Jinja2==2.11.1",
    "joblib==0.14.1",
    "jsonschema==3.2.0",
    "jupyter==1.0.0",
    "jupyter-client==6.1.2",
    "jupyter-console==6.1.0",
    "jupyter-core==4.6.3",
    "jupyter-http-over-ws==0.0.8",
    "Keras-Applications==1.0.8",
    "Keras-Preprocessing==1.1.0",
    "kiwisolver==1.2.0",
    "Markdown==3.2.1",
    "MarkupSafe==1.1.1",
    "matplotlib==3.2.1",
    "mistune==0.8.4",
    "nbconvert==5.6.1",
    "nbformat==5.0.5",
    "networkx==2.4",
    "notebook==6.0.3",
    "numpy==1.18.2",
    "oauthlib==3.1.0",
    "opencv-python==4.2.0.34",
    "opt-einsum==3.2.0",
    "pandas==1.0.3",
    "pandocfilters==1.4.2",
    "parso==0.6.2",
    "pathspec==0.8.0",
    "pexpect==4.8.0",
    "pickleshare==0.7.5",
    "Pillow==7.1.1",
    "prometheus-client==0.7.1",
    "prompt-toolkit==3.0.5",
    "protobuf==3.11.3",
    "ptyprocess==0.6.0",
    "pyasn1==0.4.8",
    "pyasn1-modules==0.2.8",
    "Pygments==2.6.1",
    "pyparsing==2.4.7",
    "pyrsistent==0.16.0",
    "python-dateutil==2.8.1",
    "pytz==2019.3",
    "PyWavelets==1.1.1",
    "pyzmq==19.0.0",
    "qtconsole==4.7.2",
    "QtPy==1.9.0",
    "regex==2020.4.4",
    "requests==2.23.0",
    "requests-oauthlib==1.3.0",
    "rsa==4.0",
    "scikit-image==0.16.2",
    "scikit-learn==0.22.2.post1",
    "scipy==1.4.1",
    "Send2Trash==1.5.0",
    "six==1.14.0",
    "tensorboard==2.1.1",
    "tensorflow==2.3.1",
    "tensorflow-estimator==2.1.0",
    "termcolor==1.1.0",
    "terminado==0.8.3",
    "testpath==0.4.4",
    "toml==0.10.0",
    "torch==1.4.0",
    "torchvision==0.5.0",
    "tornado==6.0.4",
    "traitlets==4.3.3",
    "typed-ast==1.4.1",
    "urllib3==1.25.8",
    "wcwidth==0.1.9",
    "webencodings==0.5.1",
    "Werkzeug==1.0.1",
    "widgetsnbextension==3.5.1",
    "wrapt==1.12.1",
    "zipp==3.1.0"
]

setuptools.setup(
    name="colorizexl",
    version="0.2.1",
    author="M. Linfoot, N. Freundlich",
    author_email="linfoot2@illinois.edu, norbert4@illinois.edu",
    description="Colorize and recolorize large images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nfreundlich/colorizexl",
    packages=setuptools.find_packages(exclude=('tests',)),
    package_dir={'colorizexl': './colorizexl'},
    package_data={'colorizexl': ['data/*.final']},
    install_requires=REQUIRED,
    data_files=[
                ('./data', ['./colorizexl/data/boy.png',
                            './colorizexl/data/boy_annotated.png',
                            './colorizexl/data/chair.png',
                            './colorizexl/data/chair_annotated.png']),
                ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

"""
Interesting resources checked for creating setup.py:
https://stackoverflow.com/questions/779495/python-access-data-in-package-subdirectory
https://docs.python.org/3/distutils/setupscript.html#installing-package-data
https://docs.python.org/3/distutils/sourcedist.html#manifest
https://github.com/kennethreitz/setup.py
"""