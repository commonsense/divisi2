from __future__ import with_statement
from fabric.api import *

def push():
    test()
    local('git commit -a', capture=False)
    local('git pull origin master')
    local('git push origin master')

def metapush():
    with cd('~/mmp/omcs'):
        push()
    with cd('~/mmp'):
        push()

def docs():
    with cd('doc'):
        local('make html', capture=False)

def test():
    local('nosetests -v', capture=False)
    with cd('doc'):
        local('make doctest', capture=False)

