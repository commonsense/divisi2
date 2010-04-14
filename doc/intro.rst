Quick start
===========

Once you've `built and installed <install.html>`_ Divisi, you can start
by making an AnalogySpace to reason about common sense concepts. This process
is explained in much more detail in :ref:`tutorial_aspace`.

1. Load it:

>>> from csc import divisi2
>>> matrix = divisi2.network.conceptnet_matrix('en')

2. Run the SVD:

>>> concept_axes, axis_weights, feature_axes = matrix.normalize_all().svd(k=100)

3. Get similar concepts (to 'teach'):

>>> sim_matrix = divisi2.reconstruct_similarity(concept_axes, axis_weights)
>>> sim_matrix.row_named('teach').top_items(10)

4. Predict properties (for 'trumpet'):

>>> predict_matrix = divisi2.reconstruct(concept_axes, axis_weights,
...                                      feature_axes)
>>> predict_matrix.row_named('trumpet').top_items(10)

(These things that look like `('right', 'IsA', 'pet')` are how we represent the
*features* in ConceptNet.)

5. Evaluate possible assertions:

>>> predict_matrix = divisi2.reconstruct(concept_axes, axis_weights,
...                                      feature_axes)

Is a dog a pet?

>>> predict_matrix.entry_named('dog', ('right', 'IsA', 'pet'))

Is a hammer a pet?

>>> predict_matrix.entry_named('hammer', ('right', 'IsA', 'pet'))

