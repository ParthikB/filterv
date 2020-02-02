from distutils.core import setup
setup(
  name = 'filterv',
  packages = ['filterv'],
  version = '0.0.1',
  license='MIT',
  description = "Visualizing what the filter values do to an image made easy! Just a few steps and you're good to go!",
  author = 'Parthik Bhandari',
  author_email = 'connect2parthik@gmail.com',
  url = 'https://github.com/ParthikB/filterv',
  download_url = 'https://github.com/ParthikB/filterv/archive/v0.0.1.tar.gz',
  keywords = ['filter', 'visual', 'machine learning', 'CNN', 'Convolutional', 'images', 'plot'],
  install_requires=[
          'opencv-python',
          'numpy',
          'matplotlib'
      ],
  classifiers=[  # Optional
    # How mature is this project? Common values are
    #   3 - Alpha
    #   4 - Beta
    #   5 - Production/Stable
    'Development Status :: 3 - Alpha',

    # Indicate who your project is intended for
    'Intended Audience :: Developers/Learners',

    # Pick your license as you wish
    
    # Specify the Python versions you support here. In particular, ensure
    # that you indicate whether you support Python 2, Python 3 or both.
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',

  ],
)