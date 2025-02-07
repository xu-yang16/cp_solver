from setuptools import find_packages, setup

setup(
    name="cp_solver",
    version="0.0.1",
    description="A fast and batch convex QP solver for PyTorch and NumPy.",
    author="Anonymous Authors",
    author_email="empty",
    platforms=["any"],
    license="Apache 2.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1,<2",
        "torch",
    ],
)
