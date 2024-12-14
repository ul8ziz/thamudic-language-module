from setuptools import setup, find_packages

setup(
    name="thamudic_env",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "numpy>=1.23.5",
        "opencv-python",
        "streamlit"
    ],
)
