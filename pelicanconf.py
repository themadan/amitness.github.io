#!/usr/bin/env python
# -*- coding: utf-8 -*- #

# Pelican Settings

# Basic
AUTHOR = 'Amit Chaudhary'
COPYRIGHT_NAME = AUTHOR
SITEURL = 'http://localhost:8000'
SITENAME = "Amit Chaudhary's Blog"
SITETITLE = AUTHOR
SITESUBTITLE = 'Thoughts and Writings'
SITEDESCRIPTION = '%s\'s Thoughts and Writings' % AUTHOR
ROBOTS = 'index, follow'
PATH = 'content'
TIMEZONE = 'Asia/Kathmandu'
DEFAULT_LANG = 'en'
OG_LOCALE = 'en_US'
LOCALE = 'en_US'
DATE_FORMATS = {
    'en': '%B %d, %Y',
}
USE_FOLDER_AS_CATEGORY = False
COPYRIGHT_YEAR = 2017
DEFAULT_PAGINATION = 10

# Theme Settings
SITELOGO = '/images/amit.png'
FAVICON = '/images/favicon.ico'
THEME = 'themes/Flex'
BROWSER_COLOR = '#333333'
PYGMENTS_STYLE = 'monokai'

# Feeds
FEED_DOMAIN = SITEURL
FEED_ALL_ATOM = 'feeds/all.atom.xml'
CATEGORY_FEED_ATOM = 'feeds/%s.atom.xml'
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Main Menu
MAIN_MENU = True
MENUITEMS = (('Archives', '/archives'),
             ('Categories', '/categories'),
             ('Tags', '/tags'),)

# Sidebar
LINKS = (('Projects', 'https://np.linkedin.com/in/amitify'),)
SOCIAL = (('icon-linkedin', 'https://np.linkedin.com/in/amitify'),
          ('icon-github-circled', 'https://github.com/studenton'),
          ('icon-mail', 'mailto:meamitkc@gmail.com?Subject=Contact%3A%20Amit%27s%20Blog&Body=Hi%20Amit%2C%0A%0A%0A'))

# Plugins
# See: http://docs.getpelican.com/en/latest/plugins.html
PLUGIN_PATHS = ['./pelican-plugins']
PLUGINS = ['sitemap', 'post_stats', 'share_post', 'feed_summary']

# Sitemap Settings
SITEMAP = {
    'format': 'xml',
    'priorities': {
        'articles': 0.6,
        'indexes': 0.6,
        'pages': 0.5,
    },
    'changefreqs': {
        'articles': 'monthly',
        'indexes': 'daily',
        'pages': 'monthly',
    }
}

# The static paths you want to have accessible on the output path "static"
STATIC_PATHS = ['images', 'extra']

# Extra metadata dictionaries keyed by relative path
EXTRA_PATH_METADATA = {
    'extra/custom.css': {'path': 'static/custom.css'},
    'extra/CNAME': {'path': 'CNAME'},
    'extra/robots.txt': {'path': 'robots.txt'}
}

# Custom settings
CUSTOM_CSS = 'static/custom.css'
HOME_HIDE_TAGS = True
USE_LESS = False
FEED_USE_SUMMARY = True

# Accounts
STATUSCAKE = False
DISQUS_SITENAME = "amit-chaudharys-blog"

# Formatting for URLS
ARTICLE_URL = '{slug}'
PAGE_URL = 'pages/{slug}'
CATEGORY_URL = 'category/{slug}'
TAG_URL = 'tag/{slug}'
AUTHOR_SAVE_AS = False
AUTHORS_SAVE_AS = False
