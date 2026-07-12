"""
==========================================================
Behavioural Metrics

Authors:
Peter Ohue
Gunnar Blohm
==========================================================
"""

import numpy as np


class BehaviourMetrics:

    """
    Computes behavioural statistics for one episode.
    """

    @staticmethod
    def path_length(trajectory):

        length = 0.0

        for i in range(1, len(trajectory)):

            p1 = np.array(trajectory[i - 1])

            p2 = np.array(trajectory[i])

            length += np.linalg.norm(p2 - p1)

        return length

    @staticmethod
    def mean_speed(speeds):

        return float(np.mean(speeds))

    @staticmethod
    def max_speed(speeds):

        return float(np.max(speeds))

    @staticmethod
    def success(goal):

        return bool(goal)

    @staticmethod
    def collision(hit):

        return bool(hit)