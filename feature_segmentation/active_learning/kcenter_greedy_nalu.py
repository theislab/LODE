# https://github.com/google/active-learning/blob/master/sampling_methods/kcenter_greedy.py

# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Returns points that minimize the maximum distance of any point to a center.
Implements the k-Center-Greedy method in Ozan Sener and Silvio Savarese.
A Geometric Approach to Active Learning for Convolutional Neural Networks.
https://arxiv.org/abs/1708.00489 2017 Distance metric defaults to l2 distance.
Features used to calculate distance are either raw features or if a model has
transform method then uses the output of model.transform(X).
Can be extended to a robust k centers algorithm that ignores a certain number of
outlier datapoints. Resulting centers are solution to multiple integer program.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class kCenterGreedy():
    def __init__(self, X, metric='euclidean'):
        self.features = np.vstack(X["embedding"].values)
        self.ids = X[["id"]]
        self.ids["patient_id"] = self.ids.id.str.split("_", expand=True)[0]
        self.metric = metric
        self.min_distances = None
        self.n_obs = self.features.shape[0]
        self.already_selected = []
        self.selected_patients = []

    def update_distances(self, cluster_centers):
        """Update min distances given cluster centers.
        Args:
        cluster_centers: indices of cluster centers
        """
        cluster_centers = [d for d in cluster_centers
                           if d not in self.already_selected]
        if cluster_centers:
            # Update min_distances for all examples given new cluster center.
            x = self.features[cluster_centers]
            dist_vecs = self.features - x
            dist = np.sqrt(np.sum(dist_vecs ** 2, axis = 1))
            dist = dist.reshape(-1, 1)  # reshape dist to (...,1)

            if self.min_distances is None:
                self.min_distances = dist
            else:
                self.min_distances = np.minimum(self.min_distances, dist)


    def select_batch_(self, already_selected, N):
        """
        Diversity promoting active learning method that greedily forms a batch
        to minimize the maximum distance to a cluster center among all unlabeled
        datapoints.
        Args:
        already_selected: index of datapoints already selected
        N: budget
        Returns:
        indices of points selected to minimize distance to cluster centers
        """
        print('Starting k-centers algorithm')
        new_batch = []
        for _ in range(N):
            if not self.already_selected:
                # Initialize centers with a randomly selected datapoint
                ind = np.random.choice(np.arange(self.n_obs))
            else:
                ind = np.argmax(self.min_distances)

                if np.max(self.min_distances) <= 0:
                    print("Not enough patients for budget in unlabelled pool, returning selected patients")
                    break

            # New examples should not be in already selected since those points
            # should have min_distance of zero to a cluster center.
            assert ind not in already_selected
            self.update_distances([ind])
            self.ids["distance"] = self.min_distances.reshape(-1)

            # selected patient
            selected_patient = self.ids.patient_id.iloc[ind]
            self.ids.distance.loc[self.ids.patient_id == selected_patient] = -1

            # set distance to same patient to zero, to not reelect
            self.min_distances = np.array([self.ids.distance.values.tolist()]).reshape(-1, 1)
            self.already_selected.append(ind)
            new_batch.append(ind)
        print('Maximum distance from cluster centers is %0.12f' % max(self.min_distances))

        self.already_selected = already_selected
        return new_batch, self.min_distances

    def get_ngb_uq(self, features, cluster_centers, min_dist, uq):

        ind_to_label = np.array([])

        for center in cluster_centers:
            x = features[center]
            x = x.reshape(1, -1)

            ''' Calculate distance matrix '''
            dist_vecs = self.features - x
            dist = np.sqrt(np.sum(dist_vecs ** 2, axis = 1))

            ''' Get neighbors '''
            ngb = np.argwhere(dist <= max(min_dist))  # indices where dist <= max(min_dist)
            ngb = ngb.squeeze()

            ''' Get most uncertain example in neighborhood '''
            uncertainty = uq[ngb]  # uncertainty values where dist <= max(min_dist)

            tmp = list(cluster_centers)
            tmp.remove(center)

            ''' Get indices to label '''
            if np.isscalar(uncertainty):
                _to_label = ngb

            else:
                ''' Filter 'budget' most uncertain examples '''
                sorted_index = np.argsort(uncertainty)
                _to_label = ngb[sorted_index[-1]]  # mapping back to index of uq

                ''' Check if is already selected '''
                i = -2

                while _to_label in tmp or _to_label in ind_to_label:
                    ''' Get next most uncertain example in neighborhood'''
                    _to_label = ngb[sorted_index[i]]
                    i -= 1

            ind_to_label = np.append(ind_to_label, _to_label)

        return ind_to_label
