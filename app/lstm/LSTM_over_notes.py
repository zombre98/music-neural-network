#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from app.features.MidiData import MidiData

DATASET_DIR = '../../data/maestro-v2.0.0'


# In[2]:


import pandas
import swifter

dtype = {
    'canonical_composer': 'category',
    'canonical_title': 'object',
    'split': 'category',
    'year': 'int64',
    'midi_filename': 'object',
    'audio_filename': 'object',
    'duration': 'float64'
}
dataset = pandas.read_json(os.path.join(DATASET_DIR, 'maestro-v2.0.0.json'))
dataset = dataset.astype(dtype)


# In[6]:


get_ipython().run_cell_magic('time', '', "def get_midi_data(filename):\n    return MidiData(os.path.join(DATASET_DIR, filename))\n\ndataset['midi_data'] = dataset['midi_filename'].swifter.progress_bar().allow_dask_on_strings().apply(get_midi_data)")


# In[11]:


dataset['midi_data'][0].notes.notes

