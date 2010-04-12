from __future__ import with_statement
from fabric.api import *

def push():
    local('git commit -a', capture=False)
    local('git pull')
    local('git push')

def metapush():
    with cd('~/mmp/omcs'):
        push()
    with cd('~/mmp'):
        push()

