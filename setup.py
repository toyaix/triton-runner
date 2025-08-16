import setuptools

setuptools.setup(
    name="triton_runner",
    version="0.1.9",
    author="Bob Huang",
    author_email="git@bobhuang.xyz",
    description="Triton multi-level runner, include cubin, ptx, ttgir etc.",
    package_dir={"": "src"},
    packages=setuptools.find_packages("src"),
    include_package_data=True,
)
