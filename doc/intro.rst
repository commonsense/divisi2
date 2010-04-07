Quick start
===========

Once you've `built and installed <install.html>`_ Divisi, you can start
by making an AnalogySpace to reason about common sense concepts. This process
is explained in much more detail in :ref:`tutorial_aspace`.

1. Download the `ConceptNet matrix data for English <http://conceptnet.media.mit.edu/dist/en_tuples.gz>`_.

2. Load it:

>>> from csc import divisi
>>> cnet_data = divisi2.load('en_tuples.pickle.gz')
>>> matrix = divisi2.make_sparse(cnet_data)

3. Run the SVD:

>>> concept_axes, axis_weights, feature_axes = matrix.normalize_all().svd(k=100)

4. Get similar concepts (to 'teach'):

>>> sim_matrix = divisi2.reconstruct_similarity(concept_axes, axis_weights)
>>> sim_matrix.row_named('teach').top_items(10)

5. Predict properties (for 'trumpet'):

>>> predict_matrix = divisi2.reconstruct(concept_axes, axis_weights,
...                                      feature_axes)
>>> predict_matrix.row_named('trumpet').top_items(10)

(These things that look like `('right', 'IsA', 'pet')` are how we represent the
*features* in ConceptNet.)

6. Evaluate possible assertions::

>>> predict_matrix = divisi2.reconstruct(concept_axes, axis_weights,
...                                      feature_axes)

Is a dog a pet?

>>> predict_matrix.entry_named('dog', ('right', 'IsA', 'pet'))

Is a hammer a pet?

>>> predict_matrix.entry_named('hammer', ('right', 'IsA', 'pet'))

