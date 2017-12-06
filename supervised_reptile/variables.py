"""
Tools for manipulating sets of variables.
"""

import numpy as np
import tensorflow as tf

def interpolate_vars(old_vars, new_vars, epsilon):
    """
    Interpolate between two sequences of variables.
    """
    res = []
    for old, new in zip(old_vars, new_vars):
        res.append(old + epsilon * (new - old))
    return res

def average_vars(var_seqs):
    """
    Average a sequence of variable sequences.
    """
    res = []
    for variables in zip(*var_seqs):
        res.append(np.mean(variables, axis=0))
    return res

class VariableState:
    """
    Manage the state of a set of variables.
    """
    def __init__(self, session, variables):
        self._session = session
        self._variables = variables
        self._placeholders = [tf.placeholder(v.dtype, shape=v.get_shape())
                              for v in variables]
        assigns = [tf.assign(v, p) for v, p in zip(self._variables, self._placeholders)]
        self._assign_op = tf.group(assigns)

    def export_variables(self):
        """
        Save the current variables.
        """
        return self._session.run(self._variables)

    def import_variables(self, values):
        """
        Restore the variables.
        """
        self._session.run(self._assign_op, feed_dict=dict(zip(self._variables, values)))
