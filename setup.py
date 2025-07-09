import setuptools

setuptools.setup(
    name="triton_ml_runner",
    version="0.0.1",
    author="Bob Huang",
    author_email="git@bobhuang.xyz",
    description="A multi-level Triton kernel launcher",
    package_dir={"": "src"},
    packages=setuptools.find_packages("src"),
    include_package_data=True,
)
