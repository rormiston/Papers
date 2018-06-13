#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) DeepClean Group (2017)
#
# This file is part of DeepClean.
# DeepClean is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DeepClean is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with DeepClean. If not, see <http://www.gnu.org/licenses/>.

"""Setup the DeepClean package
"""

import sys
import glob
import hashlib
import os.path
import subprocess

try:
    import setuptools
except ImportError:
    import ez_setup
    ez_setup.use_setuptools()
finally:
    from setuptools import (setup, find_packages)
    from setuptools.command import (build_py, egg_info)

from distutils.dist import Distribution
from distutils.cmd import Command
from distutils.command.clean import (clean, log, remove_tree)

# set basic metadata
PACKAGENAME = 'DeepClean'
AUTHOR = '''Rich Ormiston, Rana Adhikari, Michael Coughlin, Gabriele Vajente'''
AUTHOR_EMAIL = 'rich.ormiston@ligo.org'
LICENSE = 'GPLv3'

cmdclass = {}

# -- versioning ---------------------------------------------------------------

# import versioneer  # nopep8
# __version__ = versioneer.get_version()
# cmdclass.update(versioneer.get_cmdclass())
__version__ = 1.0

# -- dependencies -------------------------------------------------------------

# declare basic dependencies for each stage
setup_requires = [
]
install_requires = [
    'numpy>=1.11',
    'scipy>=0.16.0',
    'matplotlib>=1.4.1',
    'six>=1.5',
    'python-dateutil',
]
extras_require = {
    'hdf5': ['h5py>=1.3'],
    'root': ['root_numpy'],
    'segments': ['dqsegdb'],
    'hacr': ['pymysql'],
    'docs': ['sphinx', 'numpydoc', 'sphinx-bootstrap-theme',
             'sphinxcontrib-programoutput'],
}

# define 'all' as the intersection of all extras
extras_require['all'] = set(p for extra in extras_require.values()
                            for p in extra)

# test for OrderedDict
try:
    from collections import OrderedDict
except ImportError:
    install_requires.append('ordereddict>=1.1')

# importlib required for cli programs
try:
    from importlib import import_module
except ImportError:
    install_requires.append('importlib>=1.0.3')

# -- set test dependencies ----------------------------------------------------

setup_requires.append('pytest-runner')
tests_require = [
    'pytest>=3.1',
    'freezegun',
    'sqlparse',
]
if sys.version < '3':
    tests_require.append('mock')


# -- custom clean command -----------------------------------------------------

class DeepClean(clean):
    def run(self):
        if self.all:
            # remove dist
            if os.path.exists('dist'):
                remove_tree('dist')
            else:
                log.warn("'dist' does not exist -- can't clean it")
            # remove docs
            sphinx_dir = os.path.join(self.build_base, 'sphinx')
            if os.path.exists(sphinx_dir):
                remove_tree(sphinx_dir, dry_run=self.dry_run)
            else:
                log.warn("%r does not exist -- can't clean it", sphinx_dir)
            # remove setup eggs
            for egg in glob.glob('*.egg') + glob.glob('*.egg-info'):
                if os.path.isdir(egg):
                    remove_tree(egg, dry_run=self.dry_run)
                else:
                    log.info('removing %r' % egg)
                    os.unlink(egg)
            # remove Portfile
            portfile = 'Portfile'
            if os.path.exists(portfile) and not self.dry_run:
                log.info('removing %r' % portfile)
                os.unlink(portfile)
        clean.run(self)


cmdclass['clean'] = DeepClean


# -- build a Portfile for macports --------------------------------------------

class BuildPortfile(Command):
    """Generate a Macports Portfile for this project from the current build
    """
    description = 'Generate Macports Portfile'
    user_options = [
        ('version=', None, 'the X.Y.Z package version'),
        ('portfile=', None, 'target output file, default: \'Portfile\''),
        ('template=', None,
         'Portfile template, default: \'Portfile.template\''),
    ]

    def initialize_options(self):
        self.version = None
        self.portfile = 'Portfile'
        self.template = 'Portfile.template'
        self._template = None

    def finalize_options(self):
        from jinja2 import Template
        with open(self.template, 'r') as t:
            self._template = Template(t.read())

    def run(self):
        # get version from distribution
        if self.version is None:
            self.version = __version__
        # find dist file
        dist = os.path.join(
            'dist',
            '%s-%s.tar.gz' % (self.distribution.get_name(),
                              self.distribution.get_version()))
        # run sdist if needed
        if not os.path.isfile(dist):
            self.run_command('sdist')
        # get checksum digests
        log.info('reading distribution tarball %r' % dist)
        with open(dist, 'rb') as fobj:
            data = fobj.read()
        log.info('recovered digests:')
        digest = dict()
        digest['rmd160'] = self._get_rmd160(dist)
        for algo in [1, 256]:
            digest['sha%d' % algo] = self._get_sha(data, algo)
        for key, val in digest.iteritems():
            log.info('    %s: %s' % (key, val))
        # write finished portfile to file
        with open(self.portfile, 'w') as fport:
            fport.write(self._template.render(
                version=self.distribution.get_version(), **digest))
        log.info('portfile written to %r' % self.portfile)

    @staticmethod
    def _get_sha(data, algorithm=256):
        hash_ = getattr(hashlib, 'sha%d' % algorithm)
        return hash_(data).hexdigest()

    @staticmethod
    def _get_rmd160(filename):
        p = subprocess.Popen(['openssl', 'rmd160', filename],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        if p.returncode != 0:
            raise subprocess.CalledProcessError(err)
        else:
            return out.splitlines()[0].rsplit(' ', 1)[-1]


cmdclass['port'] = BuildPortfile
if 'port' in sys.argv:
    setup_requires.append('jinja2')


# -- find files ---------------------------------------------------------------

# Use the find_packages tool to locate all packages and modules
packagenames = find_packages()

# glob for all scripts
scripts = glob.glob(os.path.join('bin', '*'))

# -- run setup ----------------------------------------------------------------

# don't install things if just running --help
if '--help' in sys.argv or '--help-commands' in sys.argv:
    setup_requires = []

setup(name=PACKAGENAME,
      provides=[PACKAGENAME],
      version=__version__,
      description="Deep learning approach to nonlinear regression",
      long_description="""
          DeepClean is a collaboration-driven
          `Python <http://www.python.org>`_ package providing tools for
          training neural networks to perform nonlinear regression analysis
          for noise subtraction
      """,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      license=LICENSE,
      url='https://git.ligo.org/rich.ormiston/DeepClean',
      packages=packagenames,
      include_package_data=True,
      cmdclass=cmdclass,
      scripts=scripts,
      setup_requires=setup_requires,
      install_requires=install_requires,
      tests_require=tests_require,
      extras_require=extras_require,
      test_suite='DeepClean.tests',
      use_2to3=True,
      classifiers=[
          'Programming Language :: Python',
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'Intended Audience :: End Users/Desktop',
          'Intended Audience :: Developers',
          'Natural Language :: English',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Astronomy',
          'Topic :: Scientific/Engineering :: Physics',
          'Operating System :: POSIX',
          'Operating System :: Unix',
          'Operating System :: MacOS',
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
      ],
       ext_modules=[],
       requires=[],
       zip_safe=False,
      )

import os
os.system('mkdir -p dist')
os.system('mv *egg* dist/')
