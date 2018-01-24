try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

config = {
    'description': 'pnr',
    'author': 'Neil Seward, Eren Gultepe',
    'author_email': 'neil.seawrd@uoit.ca, Eren.Gultepe@uoit.ca',
    'version': '0.0.1',
    'packages': find_packages(),
    'name': 'pnr'
}

setup(**config)
