import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="detachminirocket_channel_relevance",
    version="0.0.1",
    author="Adrià Solana",
    description="Multivariate Time Series channel relevance with Detach-MiniROCKET",
    long_description_content_type="text/markdown",
    url="https://github.com/AdriaSolana/detachminirocket-channel-relevance",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD 3-Clause License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires='>=3.7',
)