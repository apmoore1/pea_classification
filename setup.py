from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(name='pea_classification',
      version='0.0.1',
      description='Financial NLP methods for performance, tone, and attribution detection',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/apmoore1/pea_classification',
      author='Andrew Moore',
      author_email='andrew.p.moore94@gmail.com',
      license='Apache License 2.0',
      install_requires=[
          'pandas==0.25.1',
          'xlrd==1.2.0',
          'xlwt==1.3.0',
          'scikit-learn==0.21.3',
          'imbalanced-learn==0.5.0',
          'requests>=2.18',
          'joblib==0.14.0',
          'spacy==2.2.1',
          'openpyxl==3.0.0'
      ],
      python_requires='>=3.6.1',
      packages=find_packages(),
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Programming Language :: Python :: 3.6'
      ])