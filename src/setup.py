import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="iphyre",
    version="0.0.1",
    author="CoRe",
    description="Benchmark for Interactive Physical Reasoning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://lishiqianhugh.github.io/IPHYRE/",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy', 'pygame', 'pymunk'
    ],
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
    ],
)
