from setuptools import setup, find_packages

setup(
    name='web_ml_recommender',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'scikit-learn',
    ],
)
