#!/usr/bin/env python
# -*- coding: utf-8 -*- #

import os
import sys

sys.path.append(os.curdir)

from pelicanconf import *

# Deployment Settings
SITEURL = 'https://amitness.com'
FEED_DOMAIN = SITEURL
RELATIVE_URLS = False
USE_LESS = False
GOOGLE_ANALYTICS = "UA-35734550-1"

# Delete the output directory, before generating new files
DELETE_OUTPUT_DIRECTORY = True

# Custom settings
GOOGLE_SITE_VERIFICATION = "PS9s0BdL6V07SPvYwyNDF6NfcpXFVZATd_o5DZqySkA"
HTTPS = True
