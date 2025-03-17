# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import itertools
import logging
import torch
import random
import detectron2.utils.comm as comm
import numpy as np
from tabulate import tabulate
from prettytable import PrettyTable
from detectron2.evaluation import DatasetEvaluator
from univlg.utils.misc import all_gather, is_dist_avail_and_initialized
from torch.nn import functional as F
from univlg.data_video.referrential_grounding_evaluator import ReferrentialGroundingEvaluator
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

import ipdb
st = ipdb.set_trace


class VQAEvaluator(DatasetEvaluator):
    def __init__(
        self,
        dataset_name,
        evaluate_detection,
        cfg=None
    ):
        self._logger = logging.getLogger(__name__)
        self.dataset_name = dataset_name
        self.cfg = cfg
        self.num_viz = 0
        self.results = []
        self.metric2scorer = {'cider': Cider(), 'bleu': Bleu(), 'meteor': Meteor(), 'rouge': Rouge()}

        self.evaluate_detection = evaluate_detection
        print(f"evaluates grounding: {self.evaluate_detection}")
        if self.evaluate_detection:
            self.detection_evaluator = ReferrentialGroundingEvaluator(
                dataset_name,
                thresholds=[0.25, 0.5, 0.75],
                topks=[1, 2, 5],
                cfg=cfg
            )

        self._cpu_device = torch.device("cpu")


    def reset(self):
        self.results = []
        self.num_viz = 0
        if self.evaluate_detection:
            self.detection_evaluator.reset()

    # called per batch
    def process(self, inputs, outputs):
        assert(len(inputs) == 1)
        for i in range(len(outputs[0])):
            self.results.append((outputs[0][i], inputs[0]['sr3d_data'][i]['answers']))

        if self.evaluate_detection:
            self.detection_evaluator.process(inputs, outputs[1])

    # called per dataset
    def evaluate(self):
        self.res = [self.results]

        if is_dist_avail_and_initialized():
            res = all_gather(self.res)
            res = list(itertools.chain(*res))[0]
            comm.synchronize()
            if self.evaluate_detection:
                detection_result = self.detection_evaluator.evaluate()
            if not comm.is_main_process():
                comm.synchronize()
                return {}
        else:
            res = self.results
            if self.evaluate_detection:
                detection_result = self.detection_evaluator.evaluate()

        gts = {}
        preds = {}
        eval_results = {}
        exact_matches = 0
        tokenizer = PTBTokenizer()

        for i in range(len(res)):
            if res[i][0] in res[i][1]:
                exact_matches += 1
            else:
                print(f"Incorrect: Predicted: {res[i][0]}, Ground Truth: {res[i][1]}")
            preds[i] = [{'caption': res[i][0]}]
            gts[i] = [{'caption': ans} for ans in res[i][1]]
        
        gts = tokenizer.tokenize(gts)
        preds = tokenizer.tokenize(preds)
        
        eval_results['exact_match'] = exact_matches / len(res)
        print(f'exact match: {np.round(exact_matches / len(res), 4)}, ')

        for metric_name in self.metric2scorer.keys():
            scores = self.metric2scorer[metric_name].compute_score(gts, preds)[1]
            if metric_name == 'bleu':
                if 'scanqa' in self.dataset_name:
                    scores = scores[0] # blue-1
                else:
                    scores = scores[-1] # bleu-4
            scores = np.array(scores)
            print(f'{metric_name}: {np.round(scores.mean(), 4)}, ', end='')
            eval_results[metric_name] = scores.mean()
        print()

        if self.evaluate_detection:
            eval_results.update(detection_result)

        if is_dist_avail_and_initialized():
            comm.synchronize()

        return eval_results
