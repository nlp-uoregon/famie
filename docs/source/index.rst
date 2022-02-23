.. trankit documentation master file, created by
   sphinx-quickstart on March 31 10:21:23 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

FaMIE's Documentation
================================================

FAMIE is a comprehensive  and efficient **active learning** (AL) toolkit for **multilingual information extraction** (IE). FAMIE is designed to address a fundamental problem in existing AL frameworks where annotators need to wait for a long time between annotation batches due to the time-consuming nature of model training and data selection at each AL iteration. With a novel `proxy AL mechanism <https://famie.readthedocs.io/en/latest/howitworks.html>`_ and the integration of our SOTA multilingual toolkit `Trankit <https://github.com/nlp-uoregon/trankit>`_, FAMIE can quickly provide users with a labeled dataset and a ready-to-use model for different IE tasks over `100 languages <https://trankit.readthedocs.io/en/latest/pkgnames.html#trainable-languages>`_.

FAMIE's github: https://github.com/nlp-uoregon/famie

FAMIE's demo website: http://nlp.uoregon.edu:9000/

.. toctree::
   :maxdepth: 2
   :caption: Introduction

   installation
   overview
   howitworks
