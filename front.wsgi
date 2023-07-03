#! /usr/bin/python3.8

import loggin
import sys
loggin.basicConfig(stream=sys.stderr)
sys.path.insert(0, '/var/www/cover/cover-art-generation/front.py')
from front.py import app as application
application.secret_key = 'deepcover'