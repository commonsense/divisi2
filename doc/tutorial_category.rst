.. _tutorial_category:

Tutorial: Working with categories
=================================

Divisi lets you express relationships between labels objects, such as the nodes
of a semantic network and their features, and to generalize from those
relationships.

Many of the operations that you can do to a single object (such as a single
node) can also be done to a set of them. A *category* is a set of labels and
their weights, and you can operate on categories similarly to how you operate
on individual objects.

When we use this idea together with common-sense knowledge, we call it an "ad
hoc category", reflecting the ability people have to reason about sets of things
such as "car, bus, train, bicycle" and generalize to whatever else should be in
the set.

Making a category
-----------------
A category is a SparseVector, generally having only a few entries. It is often
convenient to construct a category using :func:`divisi2.SparseVector.from_dict`, which takes in a dictionary, or :func:`divisi2.SparseVector.from_counts`, which takes in a list of labels. Here are some examples using both:

>>> from csc import divisi2
>>> happy = divisi2.SparseVector.from_dict({'happy': 1, 'sad': -1})
>>> transport = divisi2.SparseVector.from_counts(['car', 'bus', 'train', 'bicycle'])

Applying a matrix to a category
-------------------------------
Here's a really straightforward use of a category. You want to know what
features the items in the `transport` category have -- not with any sort of
prediction, just the things that appear literally in ConceptNet.

We load our matrix that maps concepts to features:

>>> cnet = divisi2.network.conceptnet_matrix('en')

Then we can just multiply our category of concepts by this matrix to get a
category of features.

>>> transport_features = divisi2.aligned_matrix_multiply(transport, cnet)
>>> transport_features
<SparseVector (545 of 20699 entries)>
>>> transport_features.to_dense().top_items()
[(('right', u'HasA', u'four wheel'), 5.04),
 (('right', u'UsedFor', u'transportation'), 4.93),
 (('right', u'MadeOf', u'metal'), 4.81),
 (('right', u'AtLocation', u'city'), 4.29),
 (('right', u'HasA', u'seat'), 4.25),
 (('right', u'AtLocation', u'garage'), 3.95),
 (('right', u'IsA', u'form transportation'), 3.95),
 (('right', u'AtLocation', u'street'), 3.89),
 (('left', u'PartOf', u'wheel'), 3.56),
 (('right', u'HasA', u'wheel'), 3.38)]

If we want to generalize the category using SVD, though, then what we really
want to multiply by is a ReconstructedMatrix. What kind of result we get
depends on what kind of ReconstructedMatrix we use.

Similarity to a category
........................

Suppose we use a similarity matrix from concepts to concepts. (You can read
about how to make such a matrix in :ref:`tutorial_aspace`; here we're just
going to load it from the module of pre-built examples).

>>> from csc.divisi2 import examples
>>> sim = examples.analogyspace_similarity()
>>> type(sim)
<class 'csc.divisi2.reconstructed.ReconstructedMatrix'>

Given a :class:`ReconstructedMatrix`, we can multiply a category through it --
either from the left side or the right side -- using the
:meth:`ReconstructedMatrix.left_category` and
:meth:`ReconstructedMatrix.right_category` methods.

A similarity matrix is symmetric, so the direction doesn't matter. So, for
example, let's find out what else belongs in the transport category.

>>> sim.left_category(transport).top_items()
[(u'bicycle', 3.55),
 (u'car', 3.50),
 (u'motorcycle', 3.48),
 (u'automobile', 3.48),
 (u'vehicle', 3.47),
 (u'truck', 3.47),
 (u'bus', 3.38),
 (u'bike', 3.38),
 (u'train', 3.36),
 (u'jeep', 3.30)]

Predictions from a category
...........................

If we use a reconstructed matrix mapping concepts to features, we can predict
what features these should have:

>>> predict = examples.analogyspace_predictions()
>>> predict.left_category(transport).top_items()
[(('right', u'UsedFor', u'travel'), 0.505),
 (('right', u'AtLocation', u'street'), 0.479),
 (('right', u'AtLocation', u'garage'), 0.475),
 (('right', u'AtLocation', u'city'), 0.448),
 (('right', u'UsedFor', u'transportation'), 0.440),
 (('right', u'UsedFor', u'drive'), 0.419),
 (('right', u'IsA', u'vehicle'), 0.391),
 (('right', u'HasA', u'wheel'), 0.348),
 (('right', u'HasA', u'four wheel'), 0.333),
 (('right', u'AtLocation', u'freeway'), 0.329)]

And we can apply it in the other direction, to predict what concepts should
have the list of features we generated earlier (an operation that is almost but
not quite like similarity):

>>> predict.right_category(transport_features).top_items()
[(u'car', 32.59),
 (u'vehicle', 13.09),
 (u'bicycle', 11.63),
 (u'automobile', 11.40),
 (u'drive', 9.25),
 (u'bus', 9.05),
 (u'truck', 8.36),
 (u'airplane', 8.35),
 (u'boat', 8.15),
 (u'street', 8.01)]

Spreading activation from a category
....................................

We can apply spreading activation to a category, as well.

>>> spread = examples.spreading_activation()
>>> spread.left_category(transport).top_items()
[(u'bicycle', 3.46),
 (u'bus', 3.38),
 (u'car', 3.23),
 (u'motorcycle', 3.18),
 (u'truck', 3.17),
 (u'parkway', 3.16),
 (u'train', 3.15),
 (u'move car', 3.12),
 (u'vehicle', 3.07),
 (u'bike', 3.06)]

With this, we can ask: What are the concepts most associated with happiness
(and its opposite, sadness)?

>>> spread.left_category(happy).top_items(5)
[(u'much happiness', 0.678),
 (u'score run', 0.671),
 (u'place bet', 0.663),
 (u'happy', 0.662),
 (u'great happiness', 0.646)]

>>> spread.left_category(-happy).top_items(5)
[(u'sad emotion', 0.732),
 (u'bawl', 0.732),
 (u'sob tear', 0.731),
 (u'baby sound', 0.731),
 (u'sad expression', 0.731)]

