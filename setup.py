"""Setup script for MTGTag package."""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Dependencies are defined in pyproject.toml
requirements = [
    "pandas>=1.5.0",
    "torch>=2.0.0",
    "transformers>=4.20.0",
    "datasets>=2.0.0",
    "accelerate>=0.26.0",
    "scikit-learn>=1.3.0",
    "numpy>=1.21.0",
    "tqdm>=4.64.0",
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
]

setup(
    name="mtgtag",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Magic: The Gathering Card Functional Tagging System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mtgtag",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Games/Entertainment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "mtgtag-diagnose=mtgtag.pipeline.diagnose:main",
            "mtgtag-clean=mtgtag.pipeline.clean:main",
            "mtgtag-domain-adapt=mtgtag.pipeline.domain_adapt:main",
            "mtgtag-train=mtgtag.pipeline.train:main",
            "mtgtag-optimize=mtgtag.pipeline.optimize:main",
            "mtgtag-classify=mtgtag.pipeline.classify:main",
        ],
    },
    include_package_data=True,
    package_data={
        "mtgtag": ["data/*.json", "data/*.csv"],
    },
)