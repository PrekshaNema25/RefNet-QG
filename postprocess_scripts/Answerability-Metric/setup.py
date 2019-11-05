from setuptools import find_packages
from setuptools import setup

if __name__ == '__main__':
    tests_require = [
                        'pytest',
                    ]
    setup(name='answerability-metric',
          version='1.0',
          description="Evaluate how answerable English questions are.",
          url='https://github.com/PrekshaNema25/Answerability-Metric',
          packages=find_packages(),
          install_requires=[
              'numpy',
              'scipy',
              'six',
          ],
          tests_require=tests_require,
          extras_require=dict(
              dev=tests_require,
          ),
          )
