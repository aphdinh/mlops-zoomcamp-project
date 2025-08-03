from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="seoul-bike-mlops",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A complete MLOps pipeline for Seoul Bike Sharing prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/seoul-bike-mlops",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
            "isort>=5.0",
            "pre-commit>=2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "seoul-bike-train=src.training.train:main",
            "seoul-bike-api=src.api.app:main",
        ],
    },
) 