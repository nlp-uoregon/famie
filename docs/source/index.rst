.. trankit documentation master file, created by
   sphinx-quickstart on March 31 10:21:23 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

FaMIE's Documentation
================================================

FAMIE is a comprehensive  and efficient **active learning** (AL) toolkit for **multilingual information extraction** (IE). FAMIE is designed to address a fundamental problem in existing AL frameworks where annotators need to wait for a long time between annotation batches due to the time-consuming nature of model training and data selection at each AL iteration. With a novel `proxy AL mechanism <https://famie.readthedocs.io/en/latest/howitworks.html>`_ and the integration of our SOTA multilingual toolkit `Trankit <https://github.com/nlp-uoregon/trankit>`_, **it takes FAMIE only a few hours to provide users with a labeled dataset and a ready-to-use model for different IE tasks over** `100 languages <https://trankit.readthedocs.io/en/latest/pkgnames.html#trainable-languages>`_.

FAMIE's technical paper: https://arxiv.org/pdf/2202.08316.pdf

FAMIE's github: https://github.com/nlp-uoregon/famie

FAMIE's demo website: http://nlp.uoregon.edu:9000/

Citation
========

If you use FAMIE in your research or products, please cite our following paper:

.. code-block:: bibtex

   @misc{vannguyen2022famie,
      title={FAMIE: A Fast Active Learning Framework for Multilingual Information Extraction},
      author={Nguyen, Minh Van and Ngo, Nghia Trung and Min, Bonan and Nguyen, Thien Huu},
      year={2022},
      eprint={2202.08316},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
   }

.. toctree::
   :maxdepth: 2
   :caption: Introduction

   installation
   overview
   howitworks
