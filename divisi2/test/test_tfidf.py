'''Tests for tfidf.'''
from csc import divisi2
from nose.tools import *

def test():
    '''Run the testcase from the Wikipedia article (in comments)'''
    # Consider a document containing 100 words wherein the word cow appears 3 times.
    # [specifically, let there be a document where 'cow' appears 3 times
    #  and 'moo' appears 97 times]
    doc = 0
    cow = 1
    moo = 2
    entries = [(3,  cow, doc),
               (97, moo, doc)]

    # Following the previously defined formulas, the term frequency (TF) for cow is then 0.03 (3 / 100).
    #self.assertEqual(tfidf.counts_for_document[doc], 100)
    #self.assertAlmostEqual(tfidf.tf(cow, doc), 0.03)

    # Now, assume we have 10 million documents and cow appears in one thousand of these.
    #  [specifically, let 'cow' appear in documents 0 and 10,000,000-1000+1 till 10,000,000
    for doc in xrange(10000000-1000+1,10000000):
        entries.append((1, cow, doc))

    # Then, the inverse document frequency is calculated as ln(10 000 000 / 1 000) = 9.21.
    #self.assertEqual(tfidf.num_documents, 10000000)
    #self.assertEqual(tfidf.num_docs_that_contain_term[cow], 1000)
    #self.assertAlmostEqual(tfidf.idf(cow), 9.21, 2)

    # The TF-IDF score is the product of these quantities: 0.03 * 9.21 = 0.28.
    mat = divisi2.sparse.SparseMatrix.from_entries(entries)
    tfidf = mat.normalize_tfidf()
    score = tfidf[cow, 0]
    assert_almost_equal(score, 0.28, 2)

def test_transposed():
    '''Same test, with the matrix transposed.'''
    # Consider a document containing 100 words wherein the word cow appears 3 times.
    # [specifically, let there be a document where 'cow' appears 3 times
    #  and 'moo' appears 97 times]
    doc = 0
    cow = 1
    moo = 2
    entries = [(3,  doc, cow),
               (97, doc, moo)]

    # Following the previously defined formulas, the term frequency (TF) for cow is then 0.03 (3 / 100).
    #self.assertEqual(tfidf.counts_for_document[doc], 100)
    #self.assertAlmostEqual(tfidf.tf(cow, doc), 0.03)

    # Now, assume we have 10 million documents and cow appears in one thousand of these.
    #  [specifically, let 'cow' appear in documents 0 and 10,000,000-1000+1 till 10,000,000
    for doc in xrange(10000000-1000+1,10000000):
        entries.append((1, doc, cow))

    # Then, the inverse document frequency is calculated as ln(10 000 000 / 1 000) = 9.21.
    #self.assertEqual(tfidf.num_documents, 10000000)
    #self.assertEqual(tfidf.num_docs_that_contain_term[cow], 1000)
    #self.assertAlmostEqual(tfidf.idf(cow), 9.21, 2)

    # The TF-IDF score is the product of these quantities: 0.03 * 9.21 = 0.28.
    mat = divisi2.sparse.SparseMatrix.from_entries(entries)
    tfidf = mat.normalize_tfidf(cols_are_terms=True)
    score = tfidf[0, cow]
    assert_almost_equal(score, 0.28, 2)

