from distutils.core import setup
import py2exe
import sys
sys.setrecursionlimit(3000)  # Increase the limit to a suitable value


setup(console=['app.py'])
