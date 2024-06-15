from setuptools import setup, find_packages

with open('requirements.txt') as f:
    packages = f.read().splitlines()

AUTHOR = 'Aleksandr Nevarko'
AUTHOR_EMAIL = 'anevarko@mail.ru'

if __name__ == "__main__":
    setup(
        name='stream_tools',
        version="0.0.2",
        packages=find_packages(exclude=['tests', 'scripts']),
        url='https://github.com/nkb-tech/stream-tools',
        license='',
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        description='',
        install_requires=packages,
        # tests_require=['pytest'],
        include_package_data=True,
        python_requires=">=3.9"
    )
