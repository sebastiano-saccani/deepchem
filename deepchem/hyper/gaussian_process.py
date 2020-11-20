"""
Contains class for gaussian process hyperparameter optimizations.
"""
from __future__ import division
from __future__ import unicode_literals

import logging
import numpy as np
import tempfile
import os
from deepchem.hyper.grid_search import HyperparamOpt
from deepchem.utils.evaluate import Evaluator
from deepchem.data import DiskDataset
from deepchem.molnet.run_benchmark_models import benchmark_classification, benchmark_regression
import pyGPGO
from pyGPGO.covfunc import matern32
from pyGPGO.acquisition import Acquisition
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.GPGO import GPGO

logger = logging.getLogger(__name__)


class GaussianProcessHyperparamOpt(HyperparamOpt):
  """Gaussian Process Global Optimization(GPGO)"""

  def hyperparam_search(
      self,
      hyperparam_range,
      train_dataset,
      valid_dataset,
      output_transformers,
      metric,
      direction=True,
      fit_args={},
      max_iter=20,
      hp_invalid_list=(
          'seed', 'nb_epoch', 'penalty_type', #'dropouts', 'bypass_dropouts',
          'n_pair_feat', 'fit_transformers', 'min_child_weight',
          'max_delta_step', 'subsample', 'colsample_bylevel',
          'colsample_bytree', 'reg_alpha', 'reg_lambda', 'scale_pos_weight',
          'base_score'),
      log_file='GPhypersearch.log'):
    """Perform hyperparams search using a gaussian process assumption

    params_dict include single-valued parameters being optimized,
    which should only contain int, float and list of int(float)

    parameters with names in hp_invalid_list will not be changed.

    For Molnet models, self.model_class is model name in string,
    params_dict = dc.molnet.preset_hyper_parameters.hps[self.model_class]

    Parameters
    ----------
    params_dict: dict
      dict including parameters and their initial values
      parameters not suitable for optimization can be added to hp_invalid_list
    train_dataset: dc.data.Dataset struct
      dataset used for training
    valid_dataset: dc.data.Dataset struct
      dataset used for validation(optimization on valid scores)
    output_transformers: list of dc.trans.Transformer
      transformers for evaluation
    metric: list of dc.metrics.Metric
      metric used for evaluation
    direction: bool
      maximization(True) or minimization(False)
    n_features: int
      number of input features
    n_tasks: int
      number of tasks
    max_iter: int
      number of optimization trials
    search_range: int(float)
      optimization on [initial values / search_range,
                       initial values * search_range]
    hp_invalid_list: list
      names of parameters that should not be optimized
    logfile: string
      name of log file, hyperparameters and results for each trial will be recorded

    Returns
    -------
    hyper_parameters: dict
      params_dict with all optimized values
    valid_performance_opt: float
      best performance on valid dataset

    """

    assert len(metric) == 1, 'Only use one metric'
    hp_list = list(hyperparam_range.keys())
    for hp in hp_invalid_list:
      if hp in hp_list:
        hp_list.remove(hp)

    hp_list_class = [hyperparam_range[hp][0].__class__ for hp in hp_list]
    assert set(hp_list_class) <= set([list, int, float])
    # Float or int hyper parameters(ex. batch_size, learning_rate)
    hp_list_single = [hp for i, hp in enumerate(hp_list) if hp_list_class[i] is not list]
    # List of float or int hyper parameters(ex. layer_sizes)
    hp_list_multiple = [] #structure of elements: (hp name, length of list, length of trainable params=1 or length of list)
    for i, hp in enumerate(hp_list):
      if hp_list_class[i] is list:
        if isinstance(hyperparam_range[hp][-1], int):
          hp_list_multiple.append((hp, hyperparam_range[hp][-1], 1))
        else:
          hp_list_multiple.append((hp, len(hyperparam_range[hp]), len(hyperparam_range[hp])))

    # Number of parameters
    n_param = len(hp_list_single)
    if len(hp_list_multiple) > 0:
      n_param = n_param + sum([hp[-1] for hp in hp_list_multiple])

    # Range of optimization
    param_range = []
    for hp in hp_list_single:
      param_range.append(('int' if isinstance(hyperparam_range[hp][0], int) else 'cont', hyperparam_range[hp]))
    for hp in hp_list_multiple:
      if isinstance(hyperparam_range[hp[0]][-1], int):
        param_range.append(('int' if isinstance(hyperparam_range[hp[0]][0][0], int) else 'cont',
                            hyperparam_range[hp[0]][0]))
      else:
        param_range.extend([('int' if isinstance(hyperparam_range[hp[0]][0][0], int) else 'cont', pr)
                            for pr in hyperparam_range[hp[0]]])

    # Dummy names
    param_name = [f'l{i:02d}' for i in range(n_param)]
    param = dict(zip(param_name, param_range))

    self.n_iter = 0
    self.opt_iter = -1
    self.opt_score = -np.inf
    def f(l00=0, l01=0, l02=0, l03=0, l04=0, l05=0, l06=0, l07=0, l08=0, l09=0,
          l10=0, l11=0, l12=0, l13=0, l14=0, l15=0, l16=0, l17=0, l18=0, l19=0,
          l20=0, l21=0, l22=0, l23=0, l24=0, l25=0, l26=0, l27=0, l28=0, l29=0,):
      """ Optimizing function
      Take in hyper parameter values and return valid set performances

      Parameters
      ----------
      l00~l19: int or float
        placeholders for hyperparameters being optimized,
        hyper_parameters dict is rebuilt based on input values of placeholders

      Returns:
      --------
      valid_scores: float
        valid set performances
      """
      args = locals()
      # Input hyper parameters
      i = 0
      hyper_parameters = {}
      for hp in hp_list_single:
        hyper_parameters[hp] = float(args[param_name[i]])
        if param_range[i][0] == 'int':
          hyper_parameters[hp] = int(hyper_parameters[hp])
        i = i + 1
      for hp in hp_list_multiple:
        if hp[-1] == 1:
          hyper_parameters[hp[0]] = [float(args[param_name[i]])] * hp[1]
        else:
          hyper_parameters[hp[0]] = [float(args[param_name[j]]) for j in range(i, i + hp[1])]
        if param_range[i][0] == 'int':
          hyper_parameters[hp[0]] = list(map(int, hyper_parameters[hp[0]]))
        i = i + hp[-1]

      logger.info(hyper_parameters)
      # Get the iter score
      score = self._get_score(hyper_parameters, train_dataset, valid_dataset, fit_args,
                              output_transformers, metric, log_file, self.opt_iter)
      # GPGO maximize performance by default, set performance to its negative value for minimization
      score = - score if not direction else score
      # Record the iteration if it is the optimal one
      if score > self.opt_score:
        self.opt_iter = self.n_iter
        self.opt_score = score
      self.n_iter += 1

      return score

    # GPGO optimization
    cov = matern32()
    gp = GaussianProcess(cov)
    acq = Acquisition(mode='ExpectedImprovement')
    gpgo = GPGO(gp, acq, f, param)
    logger.info(f'Max number of iteration: {max_iter}')
    gpgo.run(max_iter=max_iter)

    hp_opt, valid_performance_opt = gpgo.getResult()
    
    # Readout best hyper parameters
    i = 0
    hyperparam_opt = {}
    for hp in hp_list_single:
      hyperparam_opt[hp] = float(hp_opt[param_name[i]])
      if param_range[i][0] == 'int':
        hyperparam_opt[hp] = int(hyperparam_opt[hp])
      i = i + 1
    for hp in hp_list_multiple:
      if hp[-1] == 1:
        hyperparam_opt[hp[0]] = [float(hp_opt[param_name[i]])] * hp[1]
      else:
        hyperparam_opt[hp[0]] = [float(hp_opt[param_name[j]]) for j in range(i, i + hp[1])]
      if param_range[i][0] == 'int':
        hyperparam_opt[hp[0]] = list(map(int, hyperparam_opt[hp[0]]))
      i = i + hp[-1]

    # Return optimized hyperparameters
    return hyperparam_opt, valid_performance_opt, self.opt_iter

  def _get_score(self, hyper_parameters, train_dataset, valid_dataset, fit_args,
                 output_transformers, metric, log_file, opt_iter):
    # Log hyperparameters
    print(f'Hyperparams: {hyper_parameters}', file=log_file, flush=True)
    # Build model and get score
    if isinstance(train_dataset, (list, tuple)):
      scores = []
      for k, valid_fold in enumerate(train_dataset):
        model = self.model_class(hyper_parameters, opt_iter)
        with tempfile.TemporaryDirectory(prefix='train', suffix=None, dir=model.model_dir) as merge_dir:
          model.fit(DiskDataset.merge([data for j, data in enumerate(train_dataset) if j != k],
                                      merge_dir=merge_dir, verbose=False),
                    restore=False, **fit_args)
        score_fold = model.evaluate(valid_fold, metric, output_transformers)[metric[0].name]
        scores.append(score_fold)
        print(f' - score: {score_fold:.4f}', file=log_file, flush=True)
      scores = np.array(scores)
      weights = np.array(valid_dataset)
      W = np.sum(weights)
      score = np.sum(scores * weights) / W
      score_var = np.sum(np.square(scores - score) * weights) / W
      print(f'Score iter: {score:.4f}({np.sqrt(score_var):.4f})\n', file=log_file, flush=True)
    else:
      model = self.model_class(hyper_parameters, opt_iter)
      model.fit(train_dataset, **fit_args)
      score = model.evaluate(valid_dataset, metric, output_transformers)[metric[0].name]
      print(f'Score iter: {score:.4f}\n', file=log_file, flush=True)
    return score

