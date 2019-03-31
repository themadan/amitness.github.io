#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Theme-specific settings
SITENAME = 'Amit Chaudhary'
DOMAIN = 'http://localhost:8000'
BIO_TEXT = 'Ideas & Thoughts'

SITE_AUTHOR = 'Amit Chaudhary'
INDEX_DESCRIPTION = 'A fullstack software developer with core expertise in Python and related technologies.'

SIDEBAR_LINKS = [
    '<a href="/about/">About</a>',
    '<a href="/contact/">Contact</a>',
]

ICONS_PATH = 'images/icons'

GOOGLE_FONTS = [
    'Nunito Sans:300,700',
    'Source Code Pro',
]

SOCIAL_ICONS = [
    ('mailto:meamitkc@gmail.com', 'Email me', 'fa-envelope-o'),
    ('http://twitter.com/amitness', 'Follow me on Twitter', 'fa-twitter'),
    ('http://github.com/amitness', 'Browse my projects', 'fa-github'),
    ('https://np.linkedin.com/in/amitness', 'View Linkedin Profile', 'fa-linkedin'),
    ('/atom.xml', 'Atom Feed', 'fa-rss'),
]

THEME_COLOR = '#FF8000'

# Pelican settings
RELATIVE_URLS = True
SITEURL ='http://localhost:8000'
TIMEZONE = 'Asia/Kathmandu'
DEFAULT_DATE = 'fs'
DEFAULT_DATE_FORMAT = '%B %d, %Y'
DEFAULT_PAGINATION = False
SUMMARY_MAX_LENGTH = 42

# Theme Location
THEME = 'themes/pneumatic'

ARTICLE_URL = '{date:%Y}/{date:%m}/{slug}/'
ARTICLE_SAVE_AS = ARTICLE_URL + 'index.html'

PAGE_URL = '{slug}/'
PAGE_SAVE_AS = PAGE_URL + 'index.html'

ARCHIVES_SAVE_AS = 'archive/index.html'
YEAR_ARCHIVE_SAVE_AS = '{date:%Y}/index.html'
MONTH_ARCHIVE_SAVE_AS = '{date:%Y}/{date:%m}/index.html'

# Disable authors, categories, tags, and category pages
DIRECT_TEMPLATES = ['index', 'archives']
CATEGORY_SAVE_AS = 'category/{slug}.html'
TAG_SAVE_AS = ''

# Disable Atom feed generation
FEED_ATOM = 'atom.xml'
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = '%s.atom.xml'
TAG_FEED_ATOM = None
TAG_FEED_RSS = None
TRANSLATION_FEED_ATOM = None

TYPOGRIFY = True

MARKDOWN = {
    'extension_configs': {
        'markdown.extensions.admonition': {},
        'markdown.extensions.codehilite': {'linenums': True},
        'markdown.extensions.extra': {},
        'markdown.extensions.tables':{},
    },
    'output_format': 'html5',
}

CACHE_CONTENT = False
# DELETE_OUTPUT_DIRECTORY = True
# OUTPUT_PATH = 'develop'
PATH = 'content'

templates = ['404.html']
TEMPLATE_PAGES = {page: page for page in templates}

STATIC_PATHS = ['images', 'extra']
IGNORE_FILES = ['.DS_Store', 'pneumatic.scss', 'pygments.css', 'icomoon.css']

extras = ['CNAME', 'favicon.ico', 'robots.txt']
EXTRA_PATH_METADATA = {'extra/%s' % file: {'path': file} for file in extras}

PLUGIN_PATHS = ['./pelican-plugins']
PLUGINS = ['assets', 'neighbors', 'render_math', 'sitemap', 'share_post']
ASSET_SOURCE_PATHS = ['static']
ASSET_CONFIG = [
    ('cache', False),
    ('manifest', False),
    ('url_expire', False),
    ('versions', False),
]


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