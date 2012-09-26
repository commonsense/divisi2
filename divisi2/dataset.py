from __future__ import with_statement
from divisi2.sparse import SparseMatrix
import codecs

def movielens_titles(filebase):
    d = {}
    with open(filebase+'.item') as f:
        for line in f:
            if line.strip():
                id, title = line.split('|')[:2]
                id = int(id)
                d[id] = title
    return d

def movielens_ratings(filebase):
    titles = movielens_titles(filebase)
    seen = set()
    with open(filebase+'.data') as f:
        for line in f:
            if line.strip():
                user_id, movie_id, rating = line.split('\t')[:3]
                user_id = int(user_id)
                movie_id = int(movie_id)
                rating = float(rating)
                assert 1 <= rating <= 5
                movie_title = titles[movie_id]
                if (movie_title, user_id) not in seen:
                    seen.add((movie_title, user_id))
                    yield (rating, movie_title, user_id)

def conceptnet5_links(filename):
    def ambiguate_concept(concept):
        parts = concept.split('/')
        return '/'.join(parts[:4])
    for line in codecs.open(filename, encoding='utf-8'):
        weight, source, target = line.split('\t')
        if weight == 'Source' or weight == 'Weight':
            continue
        weight = float(weight)
        source = ambiguate_concept(source)
        target = ambiguate_concept(target)
        yield (weight, source, target)
        yield (weight, target, source)
