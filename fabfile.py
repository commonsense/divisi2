from __future__ import with_statement
from fabric.api import *

def push():
    test()
    git_dance()

def git_dance():
    local('git commit -av', capture=False)
    local('git pull origin master')
    local('git push origin master')

def metapush():
    push()
    with cd('~/mmp/omcs'):
        git_dance()
    with cd('~/mmp'):
        git_dance()

def release():
    metapush()
    local('python setup.py sdist upload')

def docs():
    with cd('doc'):
        local('make html', capture=False)

def test():
    local('nosetests -v', capture=False)
    with cd('doc'):
        local('make doctest', capture=False)

