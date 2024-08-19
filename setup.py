from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = []
with open('requirements.txt', 'rt') as f:
    for req in f.read().splitlines():
        if req.startswith('git+'):
            pkg_name = req.split('/')[-1].replace('.git', '')
            if "#egg=" in pkg_name:
                pkg_name = pkg_name.split("#egg=")[1]
            requirements.append(f'{pkg_name} @ {req}')
        else:
            requirements.append(req)

setup(
    name="pyterrier-genrerank",
    version="0.0.2",
    author="Kaustubh Dhole",
    author_email="kaustubh.dhole [AT] emory.edu",
    description="PyTerrier Interface for Generative Rerankers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/emory-irlab/pyterrier_genrank",
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.6',
    keywords=[
        "information retrieval",
        "large language models",
        "generative reranking",
    ],
)
