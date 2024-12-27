# -*- coding: utf-8 -*-


import numpy as np
import cdd
import time
import copy
import skdim
from itertools import product
# from itertools import combinations, product
# from scipy.spatial import ConvexHull, QhullError
from collections import Counter
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from scipy.stats import binomtest

# from polytope_package_v4 import CPU
# from LinearSegmentsTest_v3 import Model, Cube, get_restricted_cpu, loadModel, getLinearSegments
from LinearSegmentsTest_v3 import loadModel, Model
# from distribution_v3 import Distribution, propagate_adf, global_method, make_histo_grid, fit_into_uncert_vec
from distribution_v3 import fit_into_uncert_vec
from db_utils import softmax, regularizeCH, regularizeCDDMat, distance_to_hyperplane, in_hull, checkOverlap, propagateLinReg, regularizeCH, analyseInstances
from decision_boundaries_v2 import DecisionBoundaries
# from rolex import sample_radius
import lime
from lime.lime_tabular import LimeTabularExplainer
# from rolex import sample_radius, local_exp_rolex
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.linear_model import Lasso, lars_path, Ridge, ElasticNet, LogisticRegression


def fidelity(test_instances, inst_dbs, model, mode = 'cuboid', width = 0.2, num_test_samples = 100, samples = None):
    
    lfs_list = []
    lda_list = []
    lfs_list_bal = []
    lda_list_bal = []
    
    trivial_idx = []
    non_trivial_idx = []
    
    for i, inst in enumerate(test_instances):
        if type(samples) == type(None):
            if mode == "sphere":
                # same sampling procedure as in rolex
                # test_samples = sample_radius(inst, width/2 , num_test_samples)
                test_samples = np.array(inst) + skdim.datasets.hyperBall(n = num_test_samples, d = len(inst), radius = width/2, random_state=777) 
            elif mode == "cuboid":
                # for cuboid neighbourhood sample dimension-wise
                random_states = range(len(inst))
                test_samples = np.zeros((num_test_samples, len(inst)))
                for inst_dim in range(len(inst)):
                    dim_samps = inst[inst_dim] + skdim.datasets.hyperBall(n = num_test_samples, d = 1, radius = width/2, random_state = random_states[inst_dim])
                    dim_samps = np.reshape(dim_samps, (num_test_samples,))
                    test_samples[:, inst_dim] = dim_samps
        else: 
            test_samples = samples
        # get true model predictions
        prop_samps = model.propagate(test_samples)
        true_model_pred = np.argmax(prop_samps, axis = 1)
        
        # get predictions via linear regions
        if inst_dbs[i] == None:
            print("trivial instance: ", i)
            trivial_idx.append(i)
        else:
            non_trivial_idx.append(i)
            
            inst_dbs[i].cpu.getInputEquations()
            # inst_dbs[i].cpu.getOutputEquations()
            
            linreg_prop_samps = propagateLinReg(test_samples, inst_dbs[i].cpu)
            linreg_preds = np.argmax(linreg_prop_samps, axis = 1)
        
            acc = accuracy_score(true_model_pred, linreg_preds)
            bal_acc = balanced_accuracy_score(true_model_pred, linreg_preds)
            
            lda_list.append(acc)
            lda_list_bal.append(bal_acc)
    if len(lda_list) != 0:
        lda = sum(lda_list)/len(lda_list)
        lda_bal = sum(lda_list_bal)/len(lda_list_bal)
    else:
        lda = lda_bal = 1
    return lda, lda_bal

def sig_test(met_vals, test_met_vals, preds, test_labels, test_preds, instances = None, test_instances = None,
             max_met_diff = 0.05, max_distance = None, test_alpha = 0.05, test_p = 0.5):
    
    test_results = []
    test_pvals = []
    
    for i, met_val in enumerate(met_vals):
        pred_class = preds[i]
        
        # get indices of instances with similar metric values
        met_diff = np.abs(met_val - test_met_vals)
        rel_idx = np.where(met_diff[:, pred_class] < max_met_diff)[0]
        
        # restrict to nearby instance if a distance condition is required
        if max_distance is not None:
            inst_diff = instances[i] - test_instances
            inst_dist = np.linalg.norm(inst_diff, axis = 1)
            rel_idx_dist = np.where(inst_dist < max_distance)[0]
            rel_idx = np.intersect1d(rel_idx, rel_idx_dist)
        
        model_preds = np.array(test_preds)[rel_idx]
        true_preds = test_labels[rel_idx]
        # acc = accuracy_score(true_preds, model_preds)
        pred_sucs = np.sum(model_preds == true_preds)
        if len(rel_idx) > 0:
            test_res = binomtest(k = pred_sucs, n = len(rel_idx), p = test_p, alternative = 'greater')
        else:
            test_res = None
        
        test_results.append(test_res)
        if test_res is not None:
            test_pvals.append(test_res.pvalue)
        else:
            test_pvals.append(None)
    
    test_conf = [True if (p_val is None) or (p_val < test_alpha) else False for p_val in test_pvals]
        
    return test_results, test_pvals, test_conf

def sig_test_criteria(inst_criteria, test_inst_criteria, preds, test_labels, test_preds, instances = None, test_instances = None,
                      max_met_diff = 0.05, max_distance = None, test_alpha = 0.05, test_p = 0.5):
    
    '''
    order of criteria: 1: metric 1 val (of predicted class), 2: distance to nearest decision boundary, 
    number of nearby training instances of same class (as predicted) 3: correctly/ 4: incorrectly classfied,
    number of nearby training instances of other class (as predicted) 5: correctly/ 6: incorrectly classfied,
    '''
    test_results = []
    test_pvals = []
    rel_idx_list = []
    
    nr_classes = len(set(test_labels))
    
    for i, inst_crits in enumerate(inst_criteria):
        pred_class = preds[i]
        # get indices of instances with similar metric values
        met1_diff = np.abs(inst_crits[:nr_classes] - test_inst_criteria[:, :nr_classes])
        rel_idx_met1 = np.where(met1_diff[:, pred_class] < max_met_diff[0])[0]
        # print("near wrt metric 1: ", len(rel_idx_met1))
        
        other_crit_diff = np.abs(inst_crits[nr_classes:] - test_inst_criteria[:, nr_classes:])

        rel_idx_other_crit = np.where(np.all(other_crit_diff < max_met_diff[1:], axis = 1))[0]
        # print("near wrt other criteria: ", len(rel_idx_other_crit))
       
        rel_idx = np.intersect1d(rel_idx_met1, rel_idx_other_crit)
        # print("number of instances with suitable criteria: ", len(rel_idx))
        
        # restrict to nearby instance if a distance condition is required
        if max_distance is not None:
            inst_diff = instances[i] - test_instances
            inst_dist = np.linalg.norm(inst_diff, axis = 1)
            rel_idx_dist = np.where(inst_dist < max_distance)[0]
            rel_idx = np.intersect1d(rel_idx, rel_idx_dist)
        
        model_preds = np.array(test_preds)[rel_idx]
        true_preds = test_labels[rel_idx]
        # acc = accuracy_score(true_preds, model_preds)
        pred_sucs = np.sum(model_preds == true_preds)
        if len(rel_idx) > 0:
            test_res = binomtest(k = pred_sucs, n = len(rel_idx), p = test_p, alternative = 'greater')
        else:
            test_res = None
        
        test_results.append(test_res)
        rel_idx_list.append(rel_idx)
        if test_res is not None:
            test_pvals.append(test_res.pvalue)
        else:
            test_pvals.append(0.5)
    
    test_conf = [True if (p_val is None) or (p_val < test_alpha) else False for p_val in test_pvals]
        
    return test_pvals, test_results, test_conf, rel_idx_list
        
# Calculates how the different classes are distributed in the cuboid of given width.
#
# Input:
#   test_instances: List of all instances the metric should be applied for. For a
#                   single instance, still provide as a list [x].
#   test_labels: List of the corresponing labels.
#   model: Neural Network to be analyzed.
#   mode: Type of neighbourhood analyzed. Only one implemented yet: "cuboid".
#   width: Determines width of cuboid to be used.
#   certainty_vec: Determines the the dimensions to be used.
#   pred_triv: If True, estimates the triviality of the instance in the cuboid
#              by propagating a grid.
#
# Output:
#   metric_results: List of results. One result is given as a vector of the 
#                   distribution of classes, i.e. the metric_results[i][j] 
#                   contains the share of class j in the neighbourhood of the i-th instance.
#   all_inst_dbs: List of the decision boundaries objects in the neighbourhood
#                 of the instances.
#   gen_list: Not relevant.
#   time_dicts: List of one dict per instance, with the times the different processes
#               needed.
#   time_dict: Sum of all time_dicts.
#   trivial_predictions: Not relevant. 
#   model_predictions: List of predictions of the model for the given instances.
#   softmax_vals: List of the models softmax outputs for the given instances.
#     


def met1(test_instances, test_labels, model, mode = 'cuboid', width = 0.2, certainty_vec = None, pred_triv = True):
    
    multi_list = []
    metric_results = []
    trivial_predictions = []
    all_inst_dbs = []
    gen_list = []
    time_dict = {'linear regions': 0, 'decision boundaries': 0, 'metric 1': 0, 'predict trivial': 0}
    time_dicts = []
    softmax_vals = []
    model_predictions = []
    
    all_polys = 0
    samp_case = 0
    other_heuristic = 0
    
    if certainty_vec is None:
        certainty_vec = [None]*test_instances.shape[1]
    uncer_idx = [i for i,v in enumerate(certainty_vec) if v == None]
    cer_idx = [i for i,v in enumerate(certainty_vec) if v != None]
      
    
    
    if len(test_instances.shape) == 1:
        n = len(uncer_idx)
        test_labels = [test_labels]
        test_instances = [test_instances]
    elif len(test_instances.shape) == 2:
        n = len(uncer_idx)
        
    for inst_nr, w_inst in enumerate(test_instances):
        if inst_nr % 5 == 0:
            print("Test instance number: ", inst_nr)
        
        inst = w_inst[uncer_idx]
        # certainty_vector = [None] * len(uncer_idx)
        # certainty_vector.extend(w_inst[len(uncer_idx):])
        
        # print("In Metric 1: ", w_inst, inst, certainty_vec)
        # inst = w_inst
        # certainty_vector = certainty_vec
        
        metric_result = np.zeros(model.weights[-1].shape[0])
        time_dict_inst = {'linear regions': 0, 'decision boundaries': 0, 'metric 1': 0, 'predict trivial': 0}
        temp_pred = model.propagate([fit_into_uncert_vec(x=inst, cert_vec = certainty_vec)])[0]
        temp_class = list(temp_pred).index(max(temp_pred))
        model_predictions.append(temp_class)
        all_gens = []
        
        max_cube_data = w_inst + np.array(width)/2
        min_cube_data = w_inst - np.array(width)/2
        inst_data = np.array([max_cube_data, min_cube_data])
        inst_label = test_labels[inst_nr]
        
        if type(width) == float:
            vol_cb = width**n
        elif type(width) == list:
            vol_cb = np.prod(np.array(width))
    
        # initialize decision boundaries
        inst_dbs = DecisionBoundaries(model = model, data = inst_data, labels = inst_label)
        
        # get softmax value
        sm_val = softmax(inst_dbs.model.propagate([w_inst])[0])
        softmax_vals.append(sm_val)
        
        # predict triviality
        if pred_triv:
            start_time_pred_triv = time.time()
            triv_pred = inst_dbs.predictTrivialGridSM(sample_size = 2, cv = copy.copy(certainty_vec),
                                         unc = copy.copy(uncer_idx), cube_width = copy.copy(width), inst = copy.copy(inst))
            # trivial_predictions.append(triv_pred)
            samp_pred_flag = triv_pred[0]
            time_dict['predict trivial'] += time.time() - start_time_pred_triv
            time_dict_inst['predict trivial'] = time.time() - start_time_pred_triv
            
            if samp_pred_flag:
                time_dict_inst['metric 1'] = 0
                time_dict_inst['linear regions'] = 0
                time_dict_inst['decision boundaries'] = 0
                
                metric_result[temp_class] = 1
                metric_results.append(metric_result)
                all_inst_dbs.append(None)
                gen_list.append([])
                time_dicts.append(time_dict_inst)
                continue
            
        # get linear regions of cube around instance
        start_time_reg = time.time()
        # print(certainty_vector)
        try:
            inst_dbs.getLinearRegions(certainty_vector = certainty_vec)
        except:
            inst_dbs.getLinearRegions(certainty_vector = certainty_vec, version = 3)
        time_dict['linear regions'] += time.time() - start_time_reg
        time_dict_inst['linear regions'] = time.time() - start_time_reg
        
        # get decision boundaries
        start_ex_dbs = time.time()
        inst_dbs.getDecisionBoundariesExact(certainty_vec)
        time_dict['decision boundaries'] += time.time() - start_ex_dbs
        time_dict_inst['decision boundaries'] = time.time() - start_ex_dbs
        
        if True:
            # compute metric 1
            start_met1_time = time.time()
            db_in_poly = [(poly in inst_dbs.boundary_polytopes[0]) for poly in inst_dbs.cpu.polytopes]
            idx_mapping = {index1: inst_dbs.cpu.polytopes.index(elem1) for index1, elem1 in enumerate(inst_dbs.decision_boundary_matrices[3])}
            db_polys_indices = [idx_mapping[key] for key in idx_mapping.keys()]
            
            duplicate_polys = [item for item in db_polys_indices if db_polys_indices.count(item) > 1]
            unique_polys = [item for item in db_polys_indices if db_polys_indices.count(item) == 1]
            
            if len(duplicate_polys) > 1:
                multi_list.append(inst_nr)
            
            if any(db_in_poly):
                # non trivial case: there is a decision boundary in the cube
                for p_num, poly in enumerate(inst_dbs.cpu.polytopes):
                    all_polys += 1
                    if db_in_poly[p_num]:
                        
                        poly_verts = [inst_dbs.cpu.input_points_union[ptx] for ptx in poly.points]
                        # print(poly_verts[0])
                        if len(poly_verts[0]) == 1:
                            ch_whole_vol = np.linalg.norm(poly_verts[0]-poly_verts[1])
                        else:
                            ch_whole_vol = regularizeCH(poly_verts)[0]
                        vol_whole_pol = ch_whole_vol/vol_cb
                        # print(f"Whole Polytope number {p_num} volume: ", vol_whole_pol)
                        # case where there is at least one decision boundary in the linear region
                        if p_num in unique_polys:
                            
                            # print("one boundary case", p_num)
                            #polytope contains one single decision boundaries
                            db_id = [key for key, value in idx_mapping.items() if value == p_num][0]
                            mat_poly = cdd.Matrix(inst_dbs.decision_boundary_matrices[1][db_id].get_inequalities())
                        
                            
                            # the one decision boundary divides the polytope into two subpolytopes 
                            mat_upper = cdd.Matrix(mat_poly)
                            mat_lower = cdd.Matrix(mat_poly)
                            mat_upper.extend(inst_dbs.decision_boundary_matrices[2][db_id])
                            mat_lower.extend(-inst_dbs.decision_boundary_matrices[2][db_id])
                            
                            mat_upper.rep_type = cdd.RepType.INEQUALITY
                            mat_lower.rep_type = cdd.RepType.INEQUALITY
                            
                            try:
                                poly_upper = regularizeCDDMat(mat_upper, reg_type = 2, print_message = False)
                                gens_upper = poly_upper.get_generators()
                            except:
                                gens_upper = np.zeros((1,n+1))
                            
                            try:
                                poly_lower = regularizeCDDMat(mat_lower, reg_type = 2, print_message = False)
                                gens_lower = poly_lower.get_generators()
                            except:
                                gens_lower = np.zeros((1,n+1))
                            
                            if np.array(gens_upper).shape[0] == 0:
                                # print("empty upper polytope")
                                gens_upper = np.zeros((1,n+1))
                                # vol_t, vol_t_success = regularizeCH(np_gens_upper)
                                
                            if np.array(gens_lower).shape[0] == 0:
                                # print("empty lower polytope")
                                gens_lower = np.zeros((1,n+1))
                                # vol_t, vol_t_success = regularizeCH(np_gens_upper)
                            
                            np_gens_upper = np.array(gens_upper)[:, 1:]
                            np_gens_lower = np.array(gens_lower)[:, 1:]
    
                            all_gens.append(np_gens_upper)
                            all_gens.append(np_gens_lower)
                            middle_point_upper = np.mean(np_gens_upper, axis = 0)
                            middle_point_lower = np.mean(np_gens_lower, axis = 0)
                            
                            if certainty_vec == None:
                                class_upper = np.argmax(inst_dbs.model.propagate([middle_point_upper]), axis = 1)
                                class_lower = np.argmax(inst_dbs.model.propagate([middle_point_lower]), axis = 1)
                            else:
                                new_middle_upper = np.zeros(len(certainty_vec))
                                new_middle_lower = np.zeros(len(certainty_vec))
                                
                                for idx, val in enumerate(certainty_vec):
                                    if idx in cer_idx:
                                        new_middle_upper[idx] = val
                                        new_middle_lower[idx] = val
                                    else:
                                        new_middle_upper[idx] = middle_point_upper[uncer_idx.index(idx)]
                                        new_middle_lower[idx] = middle_point_lower[uncer_idx.index(idx)]
                               
                                class_upper = np.argmax(inst_dbs.model.propagate([new_middle_upper]), axis = 1)[0]
                                class_lower = np.argmax(inst_dbs.model.propagate([new_middle_lower]), axis = 1)[0]
                            
                            
                            if np_gens_upper.shape[1] == 1:
                                vol_upper = np.linalg.norm(np_gens_upper[1] - np_gens_upper[0])
                                upper_ch_success = True
                            else:
                                vol_upper, upper_ch_success = regularizeCH(np_gens_upper)
                            # print("upper vol: ", vol_upper/vol_cb)
                           
                            if upper_ch_success and (vol_upper/vol_cb <= vol_whole_pol):
                                metric_result[class_upper] += vol_upper/vol_cb
                                vol_lower_diff = vol_whole_pol*vol_cb - vol_upper
                                metric_result[class_lower] += vol_lower_diff/vol_cb
                            
                            else:
                                if np_gens_lower.shape[1] == 1:
                                    vol_lower = np.linalg.norm(np_gens_lower[1] - np_gens_lower[0])
                                    lower_ch_success = True
                                else:
                                    vol_lower, lower_ch_success = regularizeCH(np_gens_lower)
                                # print("lower vol: ", vol_lower/vol_cb)
                                
                                if lower_ch_success and (vol_lower/vol_cb <= vol_whole_pol):
                                    # print("lower success")
                                    metric_result[class_lower] += vol_lower/vol_cb
                                    vol_upper_diff = vol_whole_pol*vol_cb - vol_lower
                                    metric_result[class_upper] += vol_upper_diff/vol_cb
                                else:
                                    other_heuristic += 1
                                    # print("no successes: ", upper_ch_success, lower_ch_success, " or plausible volumes: ", vol_upper/vol_cb, vol_lower/vol_cb, " >= ", vol_whole_pol)
                                    classes = []
                                    for vert in poly_verts:
                                        whole_vert = np.concatenate((vert, w_inst[len(uncer_idx):]))
                                        class_vert = np.argmax(inst_dbs.model.propagate([whole_vert]), axis = 1)[0]
                                        classes.append(class_vert)
                                    rel_classes = list(set(classes))
                                    verts_cl_1 = [poly_verts[idx] for idx, vert_cl in enumerate(classes) if vert_cl == rel_classes[0]]
                                    verts_cl_2 = [poly_verts[idx] for idx, vert_cl in enumerate(classes) if vert_cl == rel_classes[1]]
                                    
                                    # use weights by volume of class polytopes
                                    vol_cl_1, cl_1_suc = regularizeCH(verts_cl_1)
                                    vol_cl_2, cl_2_suc = regularizeCH(verts_cl_2)
                                    vol_cl_1 = vol_cl_1/vol_cb
                                    vol_cl_2 = vol_cl_2/vol_cb
                                    if vol_cl_1 + vol_cl_2 > 0:
                                        sh_1 = vol_cl_1/(vol_cl_1 + vol_cl_2)
                                        sh_2 = 1 - sh_1
                                        
                                        metric_result[rel_classes[0]] += sh_1 * vol_whole_pol
                                        metric_result[rel_classes[1]] += sh_2 * vol_whole_pol
    
                        elif p_num in duplicate_polys:
                            # print(metric_result)
                            # print("multi boundary case")
                            #polytope contains multiple decision boundaries
                            
                            db_ids = [key for key, value in idx_mapping.items() if value == p_num]
                            # print("db ids: ", db_ids)
                            mat_polytope = inst_dbs.decision_boundary_matrices[1][db_ids[0]]
                            
                            mat_poly = mat_polytope.get_inequalities()
                            mat_poly_vert = mat_polytope.get_generators()
                            
                            mat_poly = cdd.Matrix(mat_poly, number_type = 'float')
                            mat_poly.rep_type = cdd.RepType.INEQUALITY
                            # print("Mat Poly ", mat_poly)
                            
                            # get volume of whole polytope with multiple boundaries (useful for checks later)
                            whole_np_gens_1 = np.array(mat_poly_vert)[:, 1:]
                            whole_poly_vol_1, whole_poly_success_1 = regularizeCH(whole_np_gens_1)
                            # print("Whole polytope volume :", whole_poly_vol_1/vol_cb)
                            
                            # NEW VARIANT
                            rel_sub_polys = [mat_poly]
                            # sub_poly_vols = []
                            
                            for db_nr, db_idx in enumerate(db_ids):
                                
                                new_rel_polys = []
                                sub_poly_vols = []
                                sub_poly_classes = []
                                
                                for rel_poly in rel_sub_polys:
                                    sub_poly_success_1 = False
                                    sub_poly_success_2 = False
                                    mat_poly_1 = rel_poly.copy()
                                    mat_poly_2 = rel_poly.copy()
                                    mat_poly_1.extend(inst_dbs.decision_boundary_matrices[2][db_idx])
                                    mat_poly_2.extend(- inst_dbs.decision_boundary_matrices[2][db_idx])
                                    mat_poly_1.rep_type = cdd.RepType.INEQUALITY
                                    mat_poly_2.rep_type = cdd.RepType.INEQUALITY
                                    
                                    sub_poly_1 = regularizeCDDMat(mat_poly_1, reg_type = 2)
                                    sub_poly_2 = regularizeCDDMat(mat_poly_2, reg_type = 2)
                                    
                                    sub_np_gens_1_ = np.array(sub_poly_1.get_generators())
                                    sub_np_gens_2_ = np.array(sub_poly_2.get_generators())
                                    
                                    if sub_np_gens_1_.shape[0] > n:
                                        sub_np_gens_1 = sub_np_gens_1_[:, 1:]
                                        sub_poly_vol_1, sub_poly_success_1 = regularizeCH(sub_np_gens_1)
                                        if sub_poly_success_1:
                                            new_rel_polys.append(mat_poly_1)
                                            sub_poly_vols.append(sub_poly_vol_1/vol_cb)
                                            mid_pt = np.mean(sub_np_gens_1, axis = 0)
                                            mid_prop = model.propagate([fit_into_uncert_vec(x = mid_pt, cert_vec = certainty_vec)])[0]
                                            sub_poly_class_1 = np.argmax(mid_prop, axis = 0)
                                            sub_poly_classes.append(sub_poly_class_1)
                                    # else:
                                    #     print(db_nr, db_idx, "degenerate case of subpoly 1. Shape is: ", sub_np_gens_1_.shape)
                                        
                                    if sub_np_gens_2_.shape[0] > n:
                                        sub_np_gens_2 = sub_np_gens_2_[:, 1:]
                                        sub_poly_vol_2, sub_poly_success_2 = regularizeCH(sub_np_gens_2)
                                        if sub_poly_success_2:
                                            new_rel_polys.append(mat_poly_2)
                                            sub_poly_vols.append(sub_poly_vol_2/vol_cb)
                                            mid_pt = np.mean(sub_np_gens_2, axis = 0)
                                            mid_prop = model.propagate([fit_into_uncert_vec(x = mid_pt, cert_vec = certainty_vec)])[0]
                                            sub_poly_class_2 = np.argmax(mid_prop, axis = 0)
                                            sub_poly_classes.append(sub_poly_class_2)                                   
                                    # else:
                                    #     print(db_nr, db_idx, "degenerate case of subpoly 2. Shape is: ", sub_np_gens_2_.shape)                            
                                rel_sub_polys = new_rel_polys
                                        
                            if abs(sum(sub_poly_vols) - whole_poly_vol_1/vol_cb) < 0.001:
                                for i, vol in enumerate(sub_poly_vols):
                                    metric_result[sub_poly_classes[i]] += vol                  
                            else:
                                samp_case += 1
                                num_samps = 1000
                                # print("sample case")
                                samp_ct = 0
                                all_samp_ct = 0
                                ineqs_raw = np.array(mat_poly)
                                A = -ineqs_raw[:, 1:]
                                b = ineqs_raw[:, 0]
                                vert_mins = np.min(whole_np_gens_1, axis = 0)
                                vert_maxs = np.max(whole_np_gens_1, axis = 0)
                                
                                rel_samps = []
                                all_samps = []
                                while samp_ct < num_samps and all_samp_ct <100000:
                                # while samp_ct < num_samps:
                                    samp = np.random.uniform(vert_mins, vert_maxs, (1, len(vert_mins)))
                                    y = np.squeeze(A @ np.transpose(samp)) - b
                                    if np.all(y <= 0):
                                        samp_ct += 1
                                        rel_samps.append(samp)
                                    else:
                                        all_samps.append(samp)
                                        all_samp_ct += 1
                                
                                if len(rel_samps) > 0:
                                    ineq_samps = np.reshape(np.array(rel_samps), (len(rel_samps), len(vert_mins)))
                                    # print("Volumen bounding box: ", np.prod(vert_maxs - vert_mins)/vol_cb)
                                    # print("Verhältnis Polytope: ", len(rel_samps)/(len(rel_samps) + len(all_samps)), " Volumen Polytope: ", whole_poly_vol_1/vol_cb)
                                    prop_samps = np.array(model.propagate(ineq_samps))
                                    prop_samps_classes = np.argmax(prop_samps, axis = 1)
                                    one_hot_classes = np.zeros_like(prop_samps)
                                    for i, line in enumerate(one_hot_classes):
                                        line[prop_samps_classes[i]] = 1
                                    class_vec = np.sum(one_hot_classes, axis = 0)
                                    class_vec = class_vec/np.sum(class_vec)
                                    metric_result += class_vec*(whole_poly_vol_1/vol_cb)
    
                            # print(metric_result, np.sum(metric_result))
                                       
                        else:
                            # print("something wrong here ", p_num)
                            # NEW VARIANT: SAMPLE AS IN MULTI CASE
                            poly_verts = [inst_dbs.cpu.input_points_union[ptx] for ptx in poly.points]
                            ch_whole_vol = regularizeCH(poly_verts)[0]
                            vol_whole_pol = ch_whole_vol/vol_cb
                            # print(f"Whole Polytope number {p_num} volume: ", vol_whole_pol)
                            
                            classes = []
                            for vert in poly_verts:
                                # print(vert.shape, len(uncer_idx))
                                whole_vert = np.concatenate((vert, w_inst[len(uncer_idx):]))
                                class_vert = np.argmax(inst_dbs.model.propagate([whole_vert]), axis = 1)[0]
                                classes.append(class_vert)
                            rel_classes = list(set(classes))
                            if len(rel_classes) == 2:
                                other_heuristic += 1
                                verts_cl_1 = [poly_verts[idx] for idx, vert_cl in enumerate(classes) if vert_cl == rel_classes[0]]
                                verts_cl_2 = [poly_verts[idx] for idx, vert_cl in enumerate(classes) if vert_cl == rel_classes[1]]
                                
                                # use weights by volume of class polytopes
                            
                                vol_cl_1, cl_1_suc = regularizeCH(verts_cl_1)
                                vol_cl_2, cl_2_suc = regularizeCH(verts_cl_2)
                                vol_cl_1 = vol_cl_1/vol_cb
                                vol_cl_2 = vol_cl_2/vol_cb
                                print(vol_cl_1, vol_cl_2)
                                if vol_cl_1 + vol_cl_2 > 0:
                                    sh_1 = vol_cl_1/(vol_cl_1 + vol_cl_2)
                                    sh_2 = 1 - sh_1
                                   
                                    metric_result[rel_classes[0]] += sh_1 * vol_whole_pol
                                    metric_result[rel_classes[1]] += sh_2 * vol_whole_pol
                    
                            elif len(rel_classes) > 2:
                                samp_case += 1
        
                                num_samps = 1000
                                samp_ct = 0
                                all_samp_ct = 0
                                polytope_vertices = [np.append(np.ones(1), j) for j in poly_verts]
                                poly_verts = np.array(polytope_vertices)
                                # ones_column = np.ones((poly_verts.shape[0], 1))
                                # poly_verts_ = np.hstack((ones_column, poly_verts))
                                poly_mat = cdd.Matrix(poly_verts)
                                poly_mat.rep_type = cdd.RepType.GENERATOR
                                cdd_poly = regularizeCDDMat(poly_mat, reg_type = 1, print_message=False)
                                mat_poly = cdd_poly.get_inequalities()
                                
                                ineqs_raw = np.array(mat_poly)
                                A = -ineqs_raw[:, 1:]
                                b = ineqs_raw[:, 0]
                                vert_mins = np.min(whole_np_gens_1, axis = 0)
                                vert_maxs = np.max(whole_np_gens_1, axis = 0)
                                
                                rel_samps = []
                                all_samps = []
                                while samp_ct < num_samps and all_samp_ct <100000:
                                # while samp_ct < num_samps:
                                    samp = np.random.uniform(vert_mins, vert_maxs, (1, len(vert_mins)))
                                    y = np.squeeze(A @ np.transpose(samp)) - b
                                    if np.all(y <= 0):
                                        samp_ct += 1
                                        rel_samps.append(samp)
                                    else:
                                        all_samps.append(samp)
                                        all_samp_ct += 1
                                
                                if len(rel_samps) > 0:
                                    ineq_samps = np.reshape(np.array(rel_samps), (len(rel_samps), len(vert_mins)))
                                    # print("Volumen bounding box: ", np.prod(vert_maxs - vert_mins)/vol_cb)
                                    # print("Verhältnis Polytope: ", len(rel_samps)/(len(rel_samps) + len(all_samps)), " Volumen Polytope: ", whole_poly_vol_1/vol_cb)
                                    prop_samps = np.array(model.propagate(ineq_samps))
                                    prop_samps_classes = np.argmax(prop_samps, axis = 1)
                                    one_hot_classes = np.zeros_like(prop_samps)
                                    for i, line in enumerate(one_hot_classes):
                                        line[prop_samps_classes[i]] = 1
                                    class_vec = np.sum(one_hot_classes, axis = 0)
                                    class_vec = class_vec/np.sum(class_vec)
                                    metric_result += class_vec*vol_whole_pol
    
                            # print(metric_result, np.sum(metric_result))
    
                            
                            
                    else:
                        # print("case: linear region without db", p_num)
                        # the linear region contains no decision boundary
                        poly_vertices = [inst_dbs.cpu.input_points_union[pt_idx] for pt_idx in poly.points]
                        poly_vertices = np.array(poly_vertices)
                        
                        mid_poly = np.mean(poly_vertices, axis = 0)
                                    
                        if certainty_vec == None:
                            poly_class = np.argmax(inst_dbs.model.propagate([mid_poly]), axis = 1)[0]
                        else:
                            db_pt = mid_poly
                            full_pt = np.zeros(len(certainty_vec))
                            for idx, val in enumerate(certainty_vec):
                                if idx in cer_idx:
                                    full_pt[idx] = val
                                else:
                                    full_pt[idx] = db_pt[uncer_idx.index(idx)]
                            poly_class = np.argmax(inst_dbs.model.propagate([full_pt]), axis = 1)[0]
                        if poly_vertices.shape[1] == 1:
                            vol_poly = np.linalg.norm(poly_vertices[0]-poly_vertices[1])
                        else:
                            vol_poly = regularizeCH(poly_vertices)[0]
                        metric_result[poly_class] += vol_poly/vol_cb
                        # all_gens.append(poly_vertices)
                        # print(metric_result)
                        
                
                
                # if len(all_gens) > 0:    
                #     all_gens = np.concatenate(all_gens, axis = 0)
                # else:
                #     all_gens = None
                
                metric_results.append(metric_result)
                # gen_list.append(all_gens)
                
            else:
                # trivial case: no decision boundary is in cube
                if certainty_vec == None:
                    class_inst = np.argmax(model.propagate([inst]), axis = 1)[0]
                else:
                    
                    full_inst = np.zeros(len(certainty_vec))                       
                    for idx, val in enumerate(certainty_vec):
                        if idx in cer_idx:
                            full_inst[idx] = val
                        else:
                            full_inst[idx] = inst[uncer_idx.index(idx)]
                    class_inst = np.argmax(model.propagate([full_inst]), axis = 1)[0]
                
                metric_result[class_inst] = 1
                metric_results.append(metric_result)
                gen_list.append([])
            
            time_dict['metric 1'] += time.time() - start_met1_time
            time_dict_inst['metric 1'] = time.time() - start_met1_time
        all_inst_dbs.append(inst_dbs)
        # all_inst_dbs.append(None)
        time_dicts.append(time_dict_inst)
        
    #print("Multi Count: ", len(multi_list), multi_list)
    #print("All polys ", all_polys, " heuristic ", other_heuristic, " sample case ", samp_case)
    # return metric_results, all_inst_dbs, gen_list, time_dicts, softmax_vals, trivial_predictions, time_pt, time_pt_lst, model_predictions
    return metric_results, all_inst_dbs, gen_list, time_dicts, time_dict, trivial_predictions, model_predictions, softmax_vals

# Modell-unabhängige Version von Metrik 2 ; Misst die Confidence nur bezogen auf umliegende Instanzen
# Der Hauptoutput ist 'vector' indem die relativen Häufigkeiten stehen
# der zweite output enthält die Zusatzinfo wie viele Instanzen in der Umgebung waren
def met2_a(data_train, y_train, instances, radius = 0.1, mode = "cube"): # Soll bedeuten 10% Der maximalen Ausprägung in jede Richtung
    results = []
    for instance in instances:
        classes_in_cube = []    
        dim = len(data_train[0])
        cube_lengths = []
        for i in range(dim): #Cube längen werden bestimmt relativ zu den minmax Ausprägungen der Attribute
            min_i = min([y[i] for y in data_train])
            max_i = max([y[i] for y in data_train])
            cube_lengths.append(radius * (max_i-min_i))
        for j in range(len(data_train)):
            in_cube = True
            if mode == "cube": #Check ob Instanz im Cube
                for i in range(dim):
                    if abs(instance[i]-data_train[j][i]) > cube_lengths[i]:
                        in_cube = False
            if mode == "circular": #Check ob Instanz in der Ellipse
                sum_of_components = 0   
                for i in range(dim):
                    sum_of_components += (data_train[j][i] - instance[i])**2/(cube_lengths[i]**2)
                if sum_of_components > 1:
                    in_cube = False
            if in_cube:
                classes_in_cube.append(y_train[j]) #Falls ja, wird die ground truth Klasse der Instanz gespeichert
        class_dict=Counter(classes_in_cube)
        vector = [] #Zählen der vorgekommenen Klassen und einfügen in einen Vektor mit relativen Häufigkeiten
        for k in range(len(set(y_train))):
            if k in class_dict.keys():
                vector.append(class_dict[k])
            else: 
                vector.append(0)
        vector = np.array(vector)
        if sum(vector) != 0:
            vector = vector * 1/sum(vector)
        results.append((vector, len(classes_in_cube)))
    return results

# Version b von Metrik 2 die auch das Modell miteinbezieht und eine ConfusionMatrix aufstellt.
def met2_b(data_train, y_train, instances, model, radius = 0.1, mode = "cube"): # Soll bedeuten 10% Der maximalen Ausprägung in jede Richtung
    results = []    
    for instance in instances:
        classes_in_cube = []    
        dim = len(data_train[0])
        cube_lengths = []
        for i in range(dim): #Cube längen werden bestimmt relativ zu den minmax Ausprägungen der Attribute
            min_i = min([y[i] for y in data_train])
            max_i = max([y[i] for y in data_train])
            cube_lengths.append(radius * (max_i-min_i))
        for j in range(len(data_train)):
            in_cube = True
            if mode == "cube": #Check ob Instanz im Cube
                for i in range(dim):
                    if abs(instance[i]-data_train[j][i]) > cube_lengths[i]:
                        in_cube = False
            if mode == "circular": #Check ob Instanz in der Ellipse
                sum_of_components = 0   
                for i in range(dim):
                    sum_of_components += (data_train[j][i] - instance[i])**2/(cube_lengths[i]**2)
                if sum_of_components > 1:
                    in_cube = False
            if in_cube: #Falls ja, wird die ground truth Klasse der Instanz, sowie deren Prediction des Modells gespeichert
                out_predicted = model.propagate([data_train[j]])[0]
                class_predicted = list(out_predicted).index(max(out_predicted))
                classes_in_cube.append((y_train[j], class_predicted))
        vector = []                         #Erstellen der Confusion Matrix. Im Format: vector[i] ist der Vektor mit den absoluten Anzahlen der 
        for k in range(len(set(y_train))):  #Predicteten Klassen für GroundTruth Klasse i Instanzen
            k_pred = []
            for a in classes_in_cube:
                if a[0] == k:
                    k_pred.append(a[1])
            k_dict = Counter(k_pred)
            k_vec = []
            for f in range(len(set(y_train))):
                if f in k_dict.keys():
                    k_vec.append(k_dict[f])
                else: 
                    k_vec.append(0)
            vector.append(k_vec)
        vector = np.array(vector) 
        results.append(vector)
    return results

if __name__ == "__main__" and 0:
    track_time = False
    global_variant = False
 
    PLATZHALTER = 'car'
    alpha = 0
    width = 0.2
    NR_FOLDS = 5
    
    # for fld in range(NR_FOLDS):
    for fld in range(1):
        FOLD = fld + 1
        print("FOLD", FOLD)
        
        model = loadModel(f"datasets/{PLATZHALTER}/model_{PLATZHALTER}_fold_{FOLD}.pt", alpha = alpha)
        with open(f"datasets/{PLATZHALTER}/data/{PLATZHALTER}_data_test_fold_{FOLD}.npy",'rb') as handle:
            all_data = np.load(handle)
            y_all_data = np.load(handle)
            
        with open(f"datasets/{PLATZHALTER}/data/{PLATZHALTER}_data_train_fold_{FOLD}.npy",'rb') as handle:
            train_data = np.load(handle)
            y_train_data = np.load(handle)
      
        test_instance = all_data[215:216]
        # test_instance = all_data[:]
        test_label = y_all_data[:]
        # cert_vec = [None, None, None, None, None, None, 0, 0, 0, 0, 0, 0, 0]
        cert_vec = None

        
        #compute metric 1 with the new local variant
        if track_time == True:
            cProfile.run('met1(test_instances = test_instance, test_labels = test_label, width = width, model = model, certainty_vec = cert_vec, reg = True, reg_eqs = 10)')
        else:
            Results = []
            new_met1 = met1(test_instances = test_instance, test_labels = test_label, width = width, 
                                model = model, certainty_vec = cert_vec, reg = False, reg_eqs = 10, pred_triv = True)
            met1_sums = [met1_res.sum() for met1_res in new_met1[0]]
            print("Metric 1 sums: ", met1_sums)