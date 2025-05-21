# -*- coding: utf-8 -*-
import datetime
import os

import numpy as np
import copy
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import cdd
import pickle
import sys
import pandas as pd
import time
from datetime import date
# from sklearn.datasets import load_breast_cancer
from itertools import combinations

from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px



import LinearSegmentsTest_v3
from polytope_package_v4 import CPU
from polytope_package_v3 import CPU_v3
from LinearSegmentsTest_v3 import Model, Cube, get_restricted_cpu, loadModel, getLinearSegments
from distribution_v3 import fit_into_uncert_vec
from db_utils import softmax, regularizeCDDMat, distance_to_hyperplane, in_hull, checkOverlap, propagateLinReg, regularizeCH, analyseInstances, closest_point_in_hull_optimization, near_train_instances
from itertools import combinations, product
import shap
import tensorflow as tf


cmap = mpl.colormaps['viridis']
alpha = 0


class DecisionBoundaries():
    # parameter model needs form as saved in getTorchModel function.
    # this is automatically the case when the model_training folder is used
    # correctly.
    # data and labels are needed for plots and area for linear regions.
    # truth_config is for plots of ground truth lines only.
    
    def __init__(self, model, data, labels, truth_config = 1):
        self.model = model
        self.data = data
        self.labels = labels
        self.truth_config = truth_config
        self.pt_class_list = None
        self.lin_reg_class_dict = {}
        self.boundary_polytopes = [[], [], [], []]
        self.cpu = None
        self.decision_boundaries = []
        self.exact_decision_boundaries = None
        self.decision_boundary_matrices = [[], [], [], []]
        
    # Compute the linear regions of a neural network. This is a necessary step before the decision boundaries can be computed.
    # The newer version 4 is faster and better for higher dimesions (i.e., > 3), the older version 3 is more stable and better for lower dimensions.
        
    def getLinearRegions(self, certainty_vector = None, version = 4):
        # get linear regions on a cube with size depending on the data
        if certainty_vector is None:
            n = self.model.weights[0].shape[1]
            cb = Cube(n)
            for i in range(n):
                cb.stretch(np.max(self.data, axis = 0)[i] - np.min(self.data, axis = 0)[i], i+1)
                cb.translate(np.min(self.data, axis = 0)[i], i+1)
    
            v = cb.points
            conn = cb.conn
            if version == 4:
                cpu_cb = CPU(v, conn)
            elif version == 3:
                cpu_cb = CPU_v3(v, conn)
            else:
                print("Wrong Version provided in getLinearRegions!")
            self.cpu = getLinearSegments(self.model, cpu_cb)
        
        else:
            uncer_idx = [i for i,v in enumerate(certainty_vector) if v == None]
            n = len(uncer_idx)
            cb = Cube(n)
            # print(self.data)
            # print(uncer_idx)
            # print(cb.points)
            # print(certainty_vector, n)
            for i in range(n):
                cb.stretch(np.max(self.data, axis = 0)[uncer_idx[i]] - np.min(self.data, axis = 0)[uncer_idx[i]], i+1)
                cb.translate(np.min(self.data, axis = 0)[uncer_idx[i]], i+1)
                # cb.stretch(np.max(self.data, axis = 0)[i] - np.min(self.data, axis = 0)[i], i+1)
                # cb.translate(np.min(self.data, axis = 0)[i], i+1)

            v = cb.points
            conn = cb.conn
            if version == 4:
                cpu_cb = CPU(v, conn)
            elif version == 3:
                cpu_cb = CPU_v3(v, conn)
            else:
                print("Wrong Version provided in getLinearRegions!")
            self.cpu = get_restricted_cpu(self.model, cpu_cb, certainty_vector)
    
    # Necessary intermediate step to compute the decision boundaries in the input space.
    # Is referenced in getDecisionBoundariesExact below.
    def getOutputBoundaries(self):
        n = self.model.weights[-1].shape[0]
        output_bounds_list = []
        indices = list(range(n))
        combinations_list = list(combinations(indices, 2))
        for comb in combinations_list:
            boundary_array = np.zeros(n)
            boundary_array[comb[0]] = 1
            boundary_array[comb[1]] = -1
            output_bounds_list.append(boundary_array)

        return output_bounds_list
    
    # Approximative (and for higher dimensions computationally expensive) alternative method to compute the decision boundaries.
    # Not needed for any applications.
    def getDecisionBoundariesEdge(self, k):
        # determine classes of all corners of linear regions
        idx_to_pt_list = copy.deepcopy(self.cpu.input_points_union)
        prop_input_points = self.model.propagate(self.cpu.input_points_union)
        prop_input_points = np.array(prop_input_points)
        input_pt_classes = np.argmax(prop_input_points, axis = 1)
        self.pt_class_list = list(input_pt_classes)
        for i, region in enumerate(self.cpu.polytopes):
            corner_classes = [self.pt_class_list[idx] for idx in region.points]
            corner_classes = np.array(corner_classes)
            comp_arr = corner_classes[0]*np.ones_like(corner_classes)
            # if all corners of a linear region yield the same class, add to class dict
            if np.array_equal(corner_classes, comp_arr):
                self.lin_reg_class_dict[i] = corner_classes[0]
            # if the corners habe different classes, subdivide further until
            else:
                del_class_list = []
                self.lin_reg_class_dict[i] = [list(np.unique(corner_classes))]
                self.boundary_polytopes[0].append(region)
                reg_del_tri = region.delaunay_triang(idx_to_pt_list)
                for del_tri in reg_del_tri:
                    del_tri_corner_classes = [self.pt_class_list[idx] for idx in del_tri.points]
                    del_tri_corner_classes = np.array(del_tri_corner_classes)
                    del_comp_arr = del_tri_corner_classes[0]*np.ones_like(del_tri_corner_classes)
                    if np.array_equal(del_tri_corner_classes, del_comp_arr):
                        del_class_list.append(del_tri_corner_classes[0])
                    else: 
                        del_class_list.append(list(np.unique(del_tri_corner_classes)))
                        self.boundary_polytopes[1].append(del_tri)
                        edgewise_simps = del_tri.edgewise(idx_to_pt_list, k = k)
                        for edge in edgewise_simps:
                            edge_corners = [idx_to_pt_list[idx] for idx in edge.points]
                            prop_edge_corners = self.model.propagate(edge_corners)
                            prop_edge_corners = np.array(prop_edge_corners)
                            edge_corner_classes = np.argmax(prop_edge_corners, axis = 1)
                            edge_corr_arr = edge_corner_classes[0]*np.ones_like(edge_corner_classes)
                            if not np.array_equal(edge_corner_classes, edge_corr_arr):
                                self.boundary_polytopes[2].append(edge)
                                edge_center = np.mean(np.array(edge_corners), axis = 0)
                                self.boundary_polytopes[3].append(edge_center)
                
                self.lin_reg_class_dict[i].append(del_class_list)
                
                    
        return del_tri, edge, idx_to_pt_list
    
    # Main function to compute the exact decision boundaries of a neural network 
    # and to save them in self.exact_decision_boundaries, which then contains the
    # points in the input space generating all decision boundary pieces.
    def getDecisionBoundariesExact(self, certainty_vector = None):
        regs = np.array([0,0])
        input_dec_boundaries = []
        self.cpu.prep_lin_seg_for_eval()
        output_dec_boundaries = self.getOutputBoundaries()
        
        for w_nr, w in enumerate(output_dec_boundaries):
            active_indices = list(np.nonzero(w)[0])
            non_active_indices = list(np.where(w == 0)[0])
            if len(self.boundary_polytopes[0]) == 0:
                
                if certainty_vector == None:
                    prop_input_points = self.model.propagate(self.cpu.input_points_union)
                else:
                    uncer_idx = [i for i,v in enumerate(certainty_vector) if v == None]
                    cer_idx = [i for i,v in enumerate(certainty_vector) if v != None]
                    
                    input_samps = np.array(self.cpu.input_points_union)
                    if len(input_samps.shape) == 1:
                        input_samps = np.reshape(input_samps, (len(input_samps),1))
                
                    sample_vec = np.zeros((len(self.cpu.input_points_union), len(certainty_vector)))
                    for idx, val in enumerate(certainty_vector):
                        if idx in cer_idx:
                            sample_vec[:,idx] = val
                        else:
                            sample_vec[:,idx] = input_samps[:, uncer_idx.index(idx)]
                        prop_input_points = self.model.propagate(sample_vec)
                
                prop_input_points = np.array(prop_input_points)
                input_pt_classes = np.argmax(prop_input_points, axis = 1)
                self.pt_class_list = list(input_pt_classes)
                for i, region in enumerate(self.cpu.polytopes):
                    corner_classes = [self.pt_class_list[idx] for idx in region.points]
                    corner_classes = np.array(corner_classes)
                    comp_arr = corner_classes[0]*np.ones_like(corner_classes)
                    # if all corners of a linear region yield the same class, add to class dict
                    if np.array_equal(corner_classes, comp_arr):
                        self.lin_reg_class_dict[i] = corner_classes[0]
                    # if the corners have different classes, subdivide further
                    else:
                        self.lin_reg_class_dict[i] = [list(np.unique(corner_classes))]
                        self.boundary_polytopes[0].append(region)
                        
            for lin_reg in self.boundary_polytopes[0]:
                A, b = lin_reg.aff_lin
                polytope_vertices = [np.append(np.ones(1), self.cpu.input_points_union[j]) for j in lin_reg.points]
                mat = cdd.Matrix(polytope_vertices, number_type='float')
                mat.rep_type = cdd.RepType.GENERATOR
                
                poly_flag = True
                while poly_flag == True:
                    try:
                        
                        
                        poly = cdd.Polyhedron(mat)
                        poly_flag = False
                    except:
                        regs[0] += 1
                        np_mat = np.array(mat)
                        np_mat_1 = np_mat[:,0]  
                        np_mat_2 = np_mat[:,1:]
                        # Regularisierung
                        np_mat_2 += (10**(-8))*np.ones(np_mat_2.shape)
                        np_mat_2 = np.column_stack((np_mat_1, np_mat_2))
                        
                        mat = cdd.Matrix(np_mat_2, number_type = 'float')
                        mat.rep_type = cdd.RepType.GENERATOR
                
                poly_ineq = poly
                
                ineqs = poly.get_inequalities()
    
                mat1 = cdd.Matrix(ineqs, number_type = 'float')
                if certainty_vector is None:
                    nai_ineqs = np.zeros((len(non_active_indices), self.model.weights[0].shape[1]+1))
                else:
                    nai_ineqs = np.zeros((len(non_active_indices), len(uncer_idx)+1))
                for j, nai in enumerate(non_active_indices):
                    w_nai = np.zeros(self.model.weights[-1].shape[0])
                    w_nai[active_indices[0]] = -1
                    w_nai[nai] = 1
                    nai_ineq = np.append(-w_nai @ b, -w_nai @ A)
                    nai_ineqs[j] = nai_ineq
                if nai_ineqs.shape[0] > 0:
                    mat1.extend(nai_ineqs)
                
                mat1.rep_type = cdd.RepType.INEQUALITY
                
                ex_rows = np.append(-w @ b, -w @ A)
                if certainty_vector is None:
                    ex_rows = ex_rows.reshape((1,self.model.weights[0].shape[1]+1))
                else:
                    ex_rows = ex_rows.reshape((1, len(uncer_idx)+1))
                mat1.extend(ex_rows, linear = True)
                mat1.rep_type = cdd.RepType.INEQUALITY
                
                poly_flag = True
    
                while poly_flag == True:
                    try:
                        poly = cdd.Polyhedron(mat1)
                        poly_flag = False
                    except:
                        regs[1] += 1
                        np_mat = np.array(mat1)
                        np_mat_1 = np_mat[:-ex_rows.shape[0]]  
                        np_mat_2 = np_mat[-ex_rows.shape[0]:]
                       
                        np_mat_2[-1] *= 1.1
                        np_mat_2 += 0.0001
                        mat1 = cdd.Matrix(np_mat_1, number_type = 'float')
                        mat1.extend(np_mat_2, linear = True)
                        mat1.rep_type = cdd.RepType.INEQUALITY
                
                
                gens = poly.get_generators()
                np_gens = np.array(gens)
                
                if np_gens.shape[0] != 0:
                    input_dec_boundaries.append((np_gens[:,1:]))
                    self.decision_boundary_matrices[0].append(poly)
                    self.decision_boundary_matrices[1].append(poly_ineq)
                    self.decision_boundary_matrices[2].append(ex_rows)
                    self.decision_boundary_matrices[3].append(lin_reg)
        
        # print(f"{int(regs[0])} type 1 regularisations, {int(regs[1])} type 2 regularisations")
        # print("Total number of linear regions: ", len(self.cpu.polytopes))
        # print("Number of linear regions with decision boundary: ", len(self.boundary_polytopes[0]))
        # print("Ratio of polytopes with decision boundary: ", len(self.boundary_polytopes[0])/len(self.cpu.polytopes)) 
        self.exact_decision_boundaries = input_dec_boundaries
  
    
    # The main method for calculating the distance of an instance x to the given decision boundaries (which have to be
    # computed first, using the method above). Uses optimization therefore the name.
    # Output: List of distances (one per decision boundary) and corresponding list of nearest points on these boundaries
    def getDistancestoallBoundariesOpt(self, x):
        if self.exact_decision_boundaries == None:
            print("No exact decision boundaries!")
        else: 
            distances = []
            nearest_points = []
            
            for boundary_pts in self.exact_decision_boundaries:
                #print('error region')
                #print(x)
                #print(boundary_pts)
                
                nearest_pt = closest_point_in_hull_optimization(x, boundary_pts)
                dist = np.linalg.norm(nearest_pt - x)
                
                distances.append(dist)
                nearest_points.append(nearest_pt)
                        
            return distances, nearest_points


    # Saves the calculated decision boundaries object for later use
    def saveBoundaries(self, dataset_name, dimension):
        db_dict = {}
        db_dict['model'] = self.model
        db_dict['data'] = self.data
        db_dict['labels'] = self.labels
        db_dict['pt_class_list'] = self.pt_class_list
        db_dict['lin_reg_class_dict'] = self.lin_reg_class_dict
        db_dict['cpu'] = self.cpu
        db_dict['edgewise decision boundaries'] = self.decision_boundaries
        db_dict['exact decision boundaries'] = self.exact_decision_boundaries
        matrix_list = [[], [], []]
        for i in range(len(self.decision_boundary_matrices[0])):
            matrix_list[0].append(np.array(self.decision_boundary_matrices[0][i].get_generators()))
            matrix_list[1].append(np.array(self.decision_boundary_matrices[1][i].get_inequalities()))
            matrix_list[2].append(self.decision_boundary_matrices[2][i])
        db_dict['decision boundary matrices'] = matrix_list
        
        with open(f"decision_boundaries_{dataset_name}_{dimension}_dim_{date.today()}.pkl", 'wb') as f:
            pickle.dump(db_dict, f)
            
    def predictTrivialGridSM(self, sample_size = 2, cv = None, unc = None, cube_width = None, inst = None):
        # samples = np.random.random(size = (sample_size, len(unc)))
        start_time = time.time()
        samples = []
        temp = []
        for i in range(len(unc)):
            temp_l = [k/sample_size for k in range(sample_size+1)]
            temp.append(temp_l)
        samples = list(product(*temp))
        sample_list = []
        classes_list = []
        softmax_list = []
        for weights in samples:
            samp = np.zeros(len(unc))
            for idx, weight in enumerate(weights):
                samp[idx] = inst[unc[idx]]-(cube_width/2)+ weight * cube_width
            temp = copy.copy(cv)
            j = 0
            for i in range(len(cv)):
                if cv[i] == None:
                    temp[i] = samp[j]
                    j += 1
            # print(("cv:",(cv, temp, inst)))
            sample_list.append(samp)
            temp_prop = self.model.propagate([temp])[0]
            temp_prop_sm = softmax(temp_prop)
            temp_class = list(temp_prop).index(max(temp_prop))
            classes_list.append(temp_class)
            softmax_list.append(temp_prop_sm)
            
        if len(set(classes_list)) < 2:
            return (True, softmax_list)
        else:
            return (False, softmax_list)   
            
    #from now, only plots        
    def initializePlot(self):
        if self.model.weights[0].shape[1] == 2 and self.model.weights[-1].shape[0] == 3:
            
            if self.truth_config == 1:
                plt.figure(figsize=(8, 6))
                plt.plot([-3+0.1*i for i in range(70)],[math.sin(-3+0.1*i)+0.2*(-3+0.1*i) for i in range(70)], '--', c="grey")
                plt.plot([-3+0.1*i for i in range(38,70)],[2.8+0.9*(-3+0.1*i) for i in range(38,70)], '--', c="grey")
                plt.plot(10*[0.8], [math.sin(0.8)+0.2*0.8+ i*0.11*(2.8+0.9*0.8-math.sin(0.8)-0.2*0.8) for i in range(10)], '--', c="grey")
                plt.plot([i*0.385 for i in range(10)], [0.1*(i*0.4) for i in range(10)], '--', c="grey")
                
            elif self.truth_config == 2:
                plt.figure(figsize=(8, 6))
                plt.plot([-3+0.1*i for i in range(39)],[math.sin(-3+0.1*i)+0.2*(-3+0.1*i) for i in range(39)], '--', c="grey")
                plt.plot([-3+0.1*i for i in range(51,70)],[math.sin(-3+0.1*i)+0.2*(-3+0.1*i) for i in range(51,70)], '--', c="grey")
                plt.plot([-3+0.1*i for i in range(70)],[2.8+0.9*(-3+0.1*i) for i in range(70)], '--', c="grey")
                plt.plot(10*[0.8], [math.sin(0.8)+0.2*0.8+ i*0.11*(2.8+0.9*0.8-math.sin(0.8)-0.2*0.8) for i in range(10)], '--', c="grey")
                plt.plot([i*0.385 for i in range(10)], [0.1*(i*0.4) for i in range(10)], '--', c="grey")
                plt.plot(10*[2.1], [0.21+ i*0.1*(math.sin(2.1)+0.2*2.1 - 0.21) for i in range(10)],'--', c="grey")
                
        elif self.model.weights[0].shape[1] == 2 and self.model.weights[-1].shape[0] == 2:
            
            plt.figure(figsize=(8, 6))
            plt.plot([-3+0.1*i for i in range(31)],[math.sin(-3+0.1*i)+0.2*(-3+0.1*i) for i in range(31)],'--', c="grey")
            plt.plot([-3+0.1*i for i in range(38, 52)],[math.sin(-3+0.1*i)+0.2*(-3+0.1*i) for i in range(38,52)],'--', c="grey")
            plt.plot([-3+0.1*i for i in range(38)],[2.8+0.9*(-3+0.1*i) for i in range(38)],'--', c="grey")
            plt.plot(10*[0.8], [math.sin(0.8)+0.2*0.8+ i*0.11*(2.8+0.9*0.8-math.sin(0.8)-0.2*0.8) for i in range(10)],'--', c="grey")
            plt.plot([i*0.385 for i in range(7)], [0.1*(i*0.4) for i in range(7)],'--', c="grey")
            plt.plot(10*[2.1], [0.21+ i*0.1*(math.sin(2.1)+0.2*2.1 - 0.21) for i in range(10)],'--', c="grey")
        
        elif self.model.weights[0].shape[1] == 2 and self.model.weights[-1].shape[0] == 6:
            plt.figure(figsize=(8, 6))
            plt.plot([-3+0.1*i for i in range(70)],[math.sin(-3+0.1*i)+0.2*(-3+0.1*i) for i in range(70)],'--', c="grey")
            plt.plot([-3+0.1*i for i in range(38, 52)],[math.sin(-3+0.1*i)+0.2*(-3+0.1*i) for i in range(38,52)],'--', c="grey")
            plt.plot([-3+0.1*i for i in range(70)],[2.8+0.9*(-3+0.1*i) for i in range(70)],'--', c="grey")
            plt.plot(10*[0.8], [math.sin(0.8)+0.2*0.8+ i*0.11*(2.8+0.9*0.8-math.sin(0.8)-0.2*0.8) for i in range(10)],'--', c="grey")
            plt.plot([i*0.385 for i in range(10)], [0.1*(i*0.4) for i in range(10)],'--', c="grey")
            plt.plot(10*[2.1], [0.21+ i*0.1*(math.sin(2.1)+0.2*2.1 - 0.21) for i in range(10)],'--', c="grey")
        
        else:
            # max_vals = np.max(self.data, axis = 0)
            # min_vals = np.min(self.data, axis = 0)
            # plt.figure(figsize=(max_vals[0] - min_vals[0], max_vals[1] - min_vals[1]))
            plt.figure(figsize=(8, 6))
        
    def plotData(self):
        self.initializePlot()
        plt.scatter(self.data[:,0],self.data[:,1], c = self.labels)
    
    def plotLinearRegions(self):
        self.initializePlot()
        region_points = np.array(self.cpu.input_points_union)
        plt.scatter(region_points[:,0], region_points[:,1], s = 5, c = 'black')
        for poly in self.cpu.polytopes:
            for cn in poly.conn:
                pt1 = self.cpu.input_points_union[cn[0]]
                pt2 = self.cpu.input_points_union[cn[1]]
                plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], linewidth = 0.25, c = 'black')

    
    def plotClassGrid(self):
        self.initializePlot()
        region_points = np.array(self.cpu.input_points_union)
        plt.scatter(region_points[:,0], region_points[:,1], s = 10, c = self.pt_class_list)
        for poly in self.cpu.polytopes:
            for cn in poly.conn:
                pt1 = self.cpu.input_points_union[cn[0]]
                pt2 = self.cpu.input_points_union[cn[1]]
                if self.pt_class_list[cn[0]] != self.pt_class_list[cn[1]]:
                    plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], linewidth = 0.5, c = 'red')
                else:
                    #Color of points and connections do not match yet
                    plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], linewidth = 0.25, color = list(cmap(self.pt_class_list))[cn[0]])
        
                    
    # parameter x can be an additional point                     
    def plotDecisionBoundaries(self, x = None):
        self.initializePlot()
        plt.scatter(np.array(self.boundary_polytopes[3])[:,0], np.array(self.boundary_polytopes[3])[:,1], s = 5, c = np.argmax(self.model.propagate(self.boundary_polytopes[3][:]), axis = 1))
        if x is not None:
            if len(x.shape) == 1:
                x = np.reshape(x, (x.shape[0], 1))
            plt.scatter(x[0, :], x[1,:], s = 10, c = 'red')
        
                
    def plotExactDecisionBoundaries(self, x = None, train_inst = None, linear_regions = False, certainty_vector = None, save_png = False, color = True, color_spaces = True, scale = False, axis_names = None):
        self.initializePlot()
        if linear_regions:
            region_points = np.array(self.cpu.input_points_union)
            plt.scatter(region_points[:,0], region_points[:,1], s = 1, c = 'grey')
            for poly in self.cpu.polytopes:
                if poly not in self.boundary_polytopes[0]:
                    for cn in poly.conn:
                        pt1 = self.cpu.input_points_union[cn[0]]
                        pt2 = self.cpu.input_points_union[cn[1]]
                        plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], '--', linewidth = 0.3, c = 'grey')
                else:
                    for cn in poly.conn:
                        pt1 = self.cpu.input_points_union[cn[0]]
                        pt2 = self.cpu.input_points_union[cn[1]]
                        if color:
                            plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], linewidth = 0.5, c = 'red')
                        else:
                            plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], '--', linewidth = 0.5,c = 'grey')
        if certainty_vector is None:
            for gens in self.exact_decision_boundaries:
                # plt.plot([gens[0][0], gens[1][0]], [gens[0][1], gens[1][1]], '--', c = "red")
                # plt.scatter(gens[:,0], gens[:,1], s = 10, c = np.argmax(self.model.propagate(gens[:]), axis = 1))
                prop_gens = self.model.propagate(gens[:])
                max_val = np.max(prop_gens[0])
                max_indices = np.where(np.isclose(prop_gens[0], max_val, atol = 1e-6))[0]
                if color:
                    plt.plot([gens[0][0], gens[1][0]], [gens[0][1], gens[1][1]], '--', c = cmap(max_indices[0]/2), dashes = (100,1))
                    plt.plot([gens[0][0], gens[1][0]], [gens[0][1], gens[1][1]], '--', c = cmap(max_indices[1]/2), dashes = (5,5))
                else:
                    plt.plot([gens[0][0], gens[1][0]], [gens[0][1], gens[1][1]], linewidth = 2, c = 'grey')
                plt.scatter(gens[:,0], gens[:,1], s = 1, c = "grey")
        else:
            uncer_idx = [i for i,v in enumerate(certainty_vector) if v == None]
            cer_idx = [i for i,v in enumerate(certainty_vector) if v != None]
            plt.title(f"Dimensions {uncer_idx[0], uncer_idx[1]}")
            if len(self.exact_decision_boundaries) > 0:
                for gens in self.exact_decision_boundaries:                    
                    input_samps = np.array(gens)
                    if len(input_samps.shape) == 1:
                        input_samps = np.reshape(input_samps, (len(input_samps),1))
                
                    cert_samps = np.reshape([certainty_vector[i] for i in cer_idx]*2, (2, len(cer_idx)))
                    sample_vec = np.zeros((2, len(certainty_vector)))
                    for idx, val in enumerate(certainty_vector):
                        if idx in cer_idx:
                            sample_vec[:,idx] = val
                        else:
                            sample_vec[:,idx] = input_samps[:, uncer_idx.index(idx)]
                    prop_gens = self.model.propagate(sample_vec)
                    plt.plot([gens[0][0], gens[1][0]], [gens[0][1], gens[1][1]], linewidth = 2, c = "grey")
                    plt.scatter(gens[:,0], gens[:,1], s = 0.5, c = np.argmax(self.model.propagate(sample_vec), axis = 1))
    
                    max_val = np.max(prop_gens[0])
                    max_indices = np.where(np.isclose(prop_gens[0], max_val, atol = 1e-6))[0]
                    if color:
                        plt.plot([gens[0][0], gens[1][0]], [gens[0][1], gens[1][1]], c = cmap(max_indices[0]/2), dashes = (100,1))
                        plt.plot([gens[0][0], gens[1][0]], [gens[0][1], gens[1][1]], c = cmap(max_indices[1]/2), dashes = (5,5))
                    else:
                        plt.plot([gens[0][0], gens[1][0]], [gens[0][1], gens[1][1]], linewidth = 2, c = 'grey')
                    plt.scatter(gens[:,0], gens[:,1], s = 1, c = "grey")
                
                if scale:
                    # specific for fetal health dataset
                    # plt.xlim(self.data[1][uncer_idx[0]], self.data[0][uncer_idx[0]])
                    # plt.ylim(bottom = self.data[1][uncer_idx[1]], top = self.data[0][uncer_idx[1]])
                    
                    with open(f"datasets/{PLATZHALTER}/data/{PLATZHALTER}.csv",'rb') as handle:
                        fetal_health_data = pd.read_csv(handle)
                    fetal_health_rel_dims = fetal_health_data[['baseline value', 'uterine_contractions', 'prolongued_decelerations', 'abnormal_short_term_variability', 'histogram_width']]
                    fetal_health_rel_dims = fetal_health_rel_dims.values
                    fetal_health_mins = np.min(fetal_health_rel_dims, axis = 0)
                    fetal_health_maxes = np.max(fetal_health_rel_dims, axis = 0)
                    fetal_health_diffs = fetal_health_maxes - fetal_health_mins
                    
                    maxs = fetal_health_maxes
                    mins = fetal_health_mins
                    def transform_x(x):
                        return 1*(x*(maxs[uncer_idx[0]]-mins[uncer_idx[0]]) + mins[uncer_idx[0]])
                    def transform_y(y):
                        return y*(maxs[uncer_idx[1]]-mins[uncer_idx[1]]) + mins[uncer_idx[1]]
                    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{transform_x(x):.1f}'))
                    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{transform_y(y):.1f}'))
                    plt.xlim(self.data[1][uncer_idx[0]], self.data[0][uncer_idx[0]])
                    plt.ylim(bottom = self.data[1][uncer_idx[1]], top = self.data[0][uncer_idx[1]])
                    plt.xticks(np.linspace(self.data[1][uncer_idx[0]], self.data[0][uncer_idx[0]], 6))
                    plt.yticks(np.linspace(self.data[1][uncer_idx[1]], self.data[0][uncer_idx[1]], 6))
                    
                    
                if axis_names is not None:
                    plt.xlabel(axis_names[uncer_idx[0]], fontsize = 13)
                    plt.ylabel(axis_names[uncer_idx[1]], fontsize = 13)

        if color_spaces:
            plot_maxs = np.max(np.array(self.cpu.input_points_union), axis = 0)
            plot_mins = np.min(np.array(self.cpu.input_points_union), axis = 0)
            grid_step = 100
            x_vals = np.linspace(plot_mins[0], plot_maxs[0], grid_step)
            y_vals = np.linspace(plot_mins[1], plot_maxs[1], grid_step)
            xx, yy = np.meshgrid(x_vals, y_vals)
            all_points = np.column_stack((xx.ravel(), yy.ravel()))
            all_pts_preds = [self.model.propagate([fit_into_uncert_vec(x = pt, cert_vec = certainty_vector)])[0] for pt in all_points]
            all_pts_preds = np.array(all_pts_preds)
            pt_classes = np.argmax(all_pts_preds, axis = 1)
            plt.scatter(all_points[:,0], all_points[:,1], s = 12, c = pt_classes[:], cmap = plt.cm.RdYlGn.reversed(), alpha = 0.2, vmin = 0, vmax = 2)
            
        if x is not None:
            if len(x.shape) == 1:
                x = np.reshape(x, (1, x.shape[0]))
            # if certainty_vector is not None:
            #     print(uncer_idx, x)
            #     plt.scatter(x[:, uncer_idx[0]], x[:, uncer_idx[1]], s = 10, c = 'red')
            # else:
            plt.scatter(x[:, 0], x[:,1], s = 100, c = 'black')
            
        if train_inst is not None:
            train_insts = np.array(train_inst[0])
            train_inst_classes = np.array(train_inst[1])
            ind_0 = np.where(train_inst_classes == 0)[0]
            ind_1 = np.where(train_inst_classes == 1)[0]
            ind_2 = np.where(train_inst_classes == 2)[0]
            if len(train_insts.shape) == 1:
                train_insts = np.reshape(train_insts, (1, train_insts.shape[0]))
            if certainty_vector is not None:
                uncer_idx = [i for i,v in enumerate(certainty_vector) if v == None]
                # print(ind_0)
                # print(train_insts[ind_0, uncer_idx[0]])
                plt.scatter(train_insts[ind_0[:], uncer_idx[0]], train_insts[ind_0[:], uncer_idx[1]], s = 50, marker = '^', c = 'black', alpha = 1)
                plt.scatter(train_insts[ind_1, uncer_idx[0]], train_insts[ind_1, uncer_idx[1]], s = 50, marker = 'x', c = 'black', alpha = 1)
                plt.scatter(train_insts[ind_2, uncer_idx[0]], train_insts[ind_2, uncer_idx[1]], s = 50, marker = 'd', c = 'black', alpha = 1)
            else:
                plt.scatter(train_insts[:, 0], train_insts[:,1], s = 100, marker = '+', c = train_inst_classes[:], cmap = plt.cm.RdYlGn.reversed())
                # plt.scatter(train_insts[:, 0], train_insts[:,1], s = 100, marker = '+', c = train_inst_classes[:], cmap = plt.cm.RdYlGn.reversed(), vmin = 0, vmax = 2)
       
        if save_png == True:
            thisDatetime = datetime.datetime.now()
            formDate = thisDatetime.strftime("%Y-%m-%d_%H-%M-%S")
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if certainty_vector is None:

                plt.savefig(f"{current_dir}/results/{PLATZHALTER}/figures/exact_decision_boundaries_{PLATZHALTER}_{model.weights[0].shape[1]}dim_{formDate}.png", dpi = 400)
                #plt.savefig(f"results/{PLATZHALTER}/figures/exact_decision_boundaries_{PLATZHALTER}_{model.weights[0].shape[1]}dim_{formDate}.png", dpi = 400)
                #plt.savefig(f"results/{PLATZHALTER}/figures/exact_decision_boundaries_{PLATZHALTER}_{model.weights[0].shape[1]}dim_{date.today()}.png", dpi = 400)
            else:
                plt.savefig(f"{current_dir}/results/{PLATZHALTER}/figures/exact_decision_boundaries_{PLATZHALTER}_{len(uncer_idx)}dim_{formDate}.png", dpi = 400)
                #plt.savefig(f"results/{PLATZHALTER}/figures/exact_decision_boundaries_{PLATZHALTER}_{len(uncer_idx)}dim_{formDate}.png", dpi = 400)
                #plt.savefig(f"results/{PLATZHALTER}/figures/exact_decision_boundaries_{PLATZHALTER}_{len(uncer_idx)}dim_{date.today()}.png", dpi = 400)


         

def model_wrapper(X, myModel: LinearSegmentsTest_v3.Model):
    predictions = myModel.propagate(X)
    return np.array(predictions)




def load_model_from_updated_path(dataset_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    the_model = loadModel(f"{current_dir}/datasets/{dataset_name}/model_{dataset_name}.pt", alpha=alpha)
    return the_model



def load_dataset_from_updated_path(dataset_name='iris'):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    with open(f"{current_dir}/datasets/{dataset_name}/data/{dataset_name}_data.npy", 'rb') as handle:
        iris_data = np.load(handle)
        iris_labels = np.load(handle)

    with open(f"{current_dir}/datasets/{dataset_name}/data/{dataset_name}_data_test.npy", 'rb') as handle:
        test_data = np.load(handle)
        y_test_data = np.load(handle)

    with open(f"{current_dir}/datasets/{dataset_name}/data/{dataset_name}_data_train.npy", 'rb') as handle:
        train_data = np.load(handle)
        y_train_data = np.load(handle)

    train_instances = train_data[:]
    train_label = y_train_data[:]

    test_instances = test_data[:]
    test_label = y_test_data[:]

    print(iris_data)
    print(iris_data.shape)
    print(iris_data[0])
    print(iris_labels[0])
    print(iris_data[3])
    print(iris_labels[3])
    print(iris_data[5])
    print(iris_labels[5])


    # metric 1 for test data: Metric 1 computes the relative amounts of volume of the
    # class subspaces in the cube (of size width) around the instances to be analyzed
    # In the met1 function the calculation of the DBs is included.
    # Together with other information, the volumes and the softmax values of the instances
    # are returned. More details are explained in metrics.py.
    #results = {}
    #test_met1 = met1(test_instances=test_instances, test_labels=test_label, width=0.2,
    #                 model=model, pred_triv=True)
    #test_met1_sums = [met1_res.sum() for met1_res in test_met1[0]]
    #print("Test Metric 1 sums: ", test_met1_sums)
    #results['Metric 1 Test Data'] = test_met1[0]
    #results['Softmax Test Data'] = test_met1[7]

    #metric_results, all_inst_dbs, gen_list, time_dicts, time_dict, trivial_predictions, model_predictions, softmax_vals = met1(test_instances=test_instances, test_labels=test_label, width=0.2,
    #                 model=model, pred_triv=True)

    return train_data, y_train_data, test_data, y_test_data, test_instances, test_label


def get_shap_values(train_data, y_train_data, test_data, y_test_data, loaded_model, test_instances= None):

    train_instances = np.array(train_data[:])
    train_label = np.array(y_train_data[:])

    if test_instances is None:
        test_instances = np.array(test_data[:])
    test_label = np.array(y_test_data[:])

    K = min(10, len(train_instances))
    sampled_train_instances = shap.sample(train_instances, K)

    explainer = shap.KernelExplainer(lambda X: model_wrapper(X, loaded_model), sampled_train_instances)
    shap_values = explainer.shap_values(test_instances)

    return shap_values, explainer.expected_value




def calculate_decision_boundaries(inst_id, width, test_instances, test_label, train_data, y_train_data, current_model, changed_instance=None, requested_dims=None):

    vis_instance = test_instances[inst_id]
    width_vec = np.array([width] * len(test_instances[0]))

    max_cube_data = vis_instance + np.array(width_vec) / 2
    min_cube_data = vis_instance - np.array(width_vec) / 2
    vis_inst_data = np.array([max_cube_data, min_cube_data])

    vis_instance = test_instances[inst_id]

    if changed_instance is not None:
        vis_instance = changed_instance

    near_train_insts = near_train_instances(train_data, y_train_data, [vis_instance], radius=width_vec / 2)

    idcs = list(range(len(vis_instance)))
    all_vis_dims = list(combinations(idcs, 2))

    tempBorders = {}
    dataframe_dict = {}
    for vis_dims in all_vis_dims:
        vis_dims = list(vis_dims)

        res_vis_inst = np.array(vis_instance)[vis_dims]
        cert_vec = [None if i in vis_dims else val for i, val in enumerate(vis_instance)]

        max_cube_data = vis_instance + np.array(width) / 2
        min_cube_data = vis_instance - np.array(width) / 2
        vis_inst_data = np.array([max_cube_data, min_cube_data])
        vis_label = test_label[inst_id]

        vis_dbs = DecisionBoundaries(model=current_model, data=vis_inst_data, labels=vis_label)
        vis_dbs.getLinearRegions(certainty_vector=cert_vec, version=3)
        vis_dbs.getDecisionBoundariesExact(certainty_vector=cert_vec)

        if cert_vec is None:
            distances = vis_dbs.getDistancestoallBoundariesOpt(vis_instance)
        else:
            distances = vis_dbs.getDistancestoallBoundariesOpt(res_vis_inst)

        if len(distances[0]) > 0:
            arg_min = np.argmin(distances[0])
            min_dist = distances[0][arg_min]
            nearest_db_pt = distances[1][arg_min]

            df = pd.DataFrame()
            df['Instance'] = res_vis_inst
            df['Nearest DB point'] = nearest_db_pt
            db_dir = nearest_db_pt - res_vis_inst
            df['Direction to nearest DB'] = db_dir
            df['overall distance'] = min_dist

            df = df.transpose()
        else:
            df = pd.DataFrame()
            df['Instance'] = res_vis_inst
            df['Nearest DB point'] = None
            #db_dir = nearest_db_pt - res_vis_inst
            df['Direction to nearest DB'] = None
            df['overall distance'] = None

            df = df.transpose()
        dataframe_dict[tuple(vis_dims)] = df
        tempBorders[tuple(vis_dims)] = vis_dbs.exact_decision_boundaries

    return [vis_instance, vis_inst_data, near_train_insts, tempBorders, dataframe_dict]




'''
def calculate_decision_boundaries2(cur_inst_id, cur_width, test_instances, test_label, train_data, y_train_data,
                                  current_model, changed_instance=None, requested_dims=None, change_feature_idx=None,
                                  changed_feature_value=None, ):
    start = time.time()
    # print('calc boundaries')
    # Runs a few exemplary calculations and visualizations for the specified instance (Instance 10 here)

    inst_id = cur_inst_id
    vis_instance = test_instances[inst_id]
    # vis_instance = [0.61111111, 0.5,        0.69491525, 0.49166667]
    vis_label = test_label[inst_id]
    width = cur_width

    # vis_dims = [1, 2, 3, 0]  # Dimensions to be analyzed
    # vis_dims = [0,1,2,4,5]
    vis_dims = [i for i in range(len(vis_instance))]
    if requested_dims is not None:
        vis_dims = requested_dims

    # print(f'width: {width} - test_instances[0]: {test_instances[0]}')
    # print(f'np arry: {[width]}')
    width_vec = np.array([width] * len(test_instances[0]))
    res_vis_inst = np.array(vis_instance)[vis_dims]

    # set whole cert_vec = None only if all dimensions are analyzed
    # use the definition above for subspaces
    cert_vec = None
    # cert_vec = [None if i in vis_dims else val for i, val in enumerate(vis_instance)]
    if requested_dims is not None:
        cert_vec = [None if i in vis_dims else val for i, val in enumerate(vis_instance)]
    # cert_vec = [None if i in vis_dims else val for i, val in enumerate(vis_instance)]
    max_cube_data = vis_instance + np.array(width_vec) / 2
    min_cube_data = vis_instance - np.array(width_vec) / 2
    vis_inst_data = np.array([max_cube_data, min_cube_data])
    # print(vis_dims)
    # print(cert_vec)

    # Initialize the DecisionBoundaries Object
    print('initialize with this:')
    #print(vis_inst_data)
    print(vis_inst_data, vis_label, cert_vec)
    print('#################')
    vis_dbs = DecisionBoundaries(model=current_model, data=vis_inst_data, labels=vis_label)
    print("Start Linear Regions")
    # vis_dbs.getLinearRegions(certainty_vector = cert_vec, version = 3)
    vis_dbs.getLinearRegions(certainty_vector=cert_vec)
    print("Start Decision Boundaries")
    vis_dbs.getDecisionBoundariesExact(certainty_vector=cert_vec)
    print('exact bonds')
    print(vis_dbs.exact_decision_boundaries)



    if cert_vec is None:
        distances = vis_dbs.getDistancestoallBoundariesOpt(vis_instance)
    else:
        distances = vis_dbs.getDistancestoallBoundariesOpt(res_vis_inst)

    if len(distances[0]) > 0:
        arg_min = np.argmin(distances[0])
        min_dist = distances[0][arg_min]
        nearest_db_pt = distances[1][arg_min]

        df = pd.DataFrame()
        df['Instance'] = res_vis_inst
        df['Nearest DB point'] = nearest_db_pt
        db_dir = nearest_db_pt - res_vis_inst
        df['Direction to nearest DB'] = db_dir
        df['overall distance'] = min_dist
        df = df.transpose()
        print(df)

    # df.to_excel(f'results/{PLATZHALTER}/min_distances_and_nearest_db_pt_test_instance_{inst_id}.xlsx')
    # print("Min distance to DB: ", min_dist)
    # print("Nearest DB point: ", nearest_db_pt)
    # print("Direction to DB: ", db_dir)
    # print("Normed Direction to DB: ", db_dir / np.linalg.norm(db_dir))

    # Visualize an the area around an instance in two dimensions
    # using the plot functions above. All possible two dimensional subspaces
    # are visualiszed using the respective certainty vector (cert_vec) below.
    # change_idx = 2
    # change_factor = 1.06
    vis_instance = test_instances[inst_id]
    if change_feature_idx is not None:
        vis_instance[change_feature_idx] = vis_instance[change_feature_idx] * changed_feature_value
    # vis_instance = [0.61111111, 0.5,        0.69491525, 0.49166667]

    if changed_instance is not None:
        vis_instance = changed_instance

    near_train_insts = near_train_instances(train_data, y_train_data, [vis_instance], radius=width_vec / 2)

    idcs = list(range(len(vis_instance)))
    all_vis_dims = list(combinations(idcs, 2))
    mid_1 = time.time()

    tempBorders = {}
    dataframe_dict = {}
    temp = 1
    for vis_dims in all_vis_dims:
        vis_dims = list(vis_dims)
        # print('loop with')
        # print(vis_dims)
        # print(vis_instance)

        res_vis_inst = np.array(vis_instance)[vis_dims]
        cert_vec = [None if i in vis_dims else val for i, val in enumerate(vis_instance)]
        # if cert_vec[3] is not None:
        #    cert_vec[3] = cert_vec[3] * 1.5
        max_cube_data = vis_instance + np.array(width) / 2
        min_cube_data = vis_instance - np.array(width) / 2
        vis_inst_data = np.array([max_cube_data, min_cube_data])
        vis_label = test_label[inst_id]

        # print('here')
        # print(vis_inst_data)
        # print(cert_vec)
        # if cert_vec[2] != None:
        #    cert_vec[2] = 0.64400
        # if cert_vec[3] != None:
        #    cert_vec[3] = 0.74400
        # cert_vec[3] = 0.59166
        # print(cert_vec)
        # print('#############')
        vis_dbs = DecisionBoundaries(model=current_model, data=vis_inst_data, labels=vis_label)
        vis_dbs.getLinearRegions(certainty_vector=cert_vec, version=3)
        vis_dbs.getDecisionBoundariesExact(certainty_vector=cert_vec)


        vis_dbs = DecisionBoundaries(model=current_model, data=vis_inst_data, labels=vis_label)
        print("Start Linear Regions")
        # vis_dbs.getLinearRegions(certainty_vector = cert_vec, version = 3)
        vis_dbs.getLinearRegions(certainty_vector=cert_vec)
        print("Start Decision Boundaries")
        vis_dbs.getDecisionBoundariesExact(certainty_vector=cert_vec)


        if cert_vec is None:
            distances = vis_dbs.getDistancestoallBoundariesOpt(vis_instance)
        else:
            distances = vis_dbs.getDistancestoallBoundariesOpt(res_vis_inst)

        if len(distances[0]) > 0:
            arg_min = np.argmin(distances[0])
            min_dist = distances[0][arg_min]
            nearest_db_pt = distances[1][arg_min]

            df = pd.DataFrame()
            df['Instance'] = res_vis_inst
            df['Nearest DB point'] = nearest_db_pt
            db_dir = nearest_db_pt - res_vis_inst
            df['Direction to nearest DB'] = db_dir
            df['overall distance'] = min_dist

            df = df.transpose()
        else:
            df = pd.DataFrame()
            df['Instance'] = res_vis_inst
            df['Nearest DB point'] = None
            # db_dir = nearest_db_pt - res_vis_inst
            df['Direction to nearest DB'] = None
            df['overall distance'] = None

            df = df.transpose()
        dataframe_dict[tuple(vis_dims)] = df
        # print(vis_dims)
        # print(df)
        # df.to_excel(f'results/{PLATZHALTER}/min_distances_and_nearest_db_pt_test_instance_{inst_id}.xlsx')
        # print("Min distance to DB: ", min_dist)
        # print("Nearest DB point: ", nearest_db_pt)
        # print("Direction to DB: ", db_dir)
        # print("Normed Direction to DB: ", db_dir / np.linalg.norm(db_dir))

        # print('res_vis_inst')
        # print(res_vis_inst)
        tempBorders[tuple(vis_dims)] = vis_dbs.exact_decision_boundaries

        # tempBorders.append(vis_dbs.exact_decision_boundaries)
        # distances = vis_dbs.getDistancestoallBoundariesOpt(res_vis_inst)
        # print("Minimum Distance to a DB in the Subspace: ", np.min(distances[0]))

        # vis_dbs.plotExactDecisionBoundaries(x=res_vis_inst, train_inst=near_train_insts, linear_regions=False,
        #                                    certainty_vector=cert_vec,
        #                                    color=False, color_spaces=True, save_png=True, scale=False)

    return [vis_instance, vis_inst_data, near_train_insts, tempBorders, dataframe_dict]



def calc_boundaries_2(current_model, test_instances, train_data, y_train_data, cur_width, vis_instance, vis_label):
    start = time.time()
    # print('calc boundaries')
    # Runs a few exemplary calculations and visualizations for the specified instance (Instance 10 here)

    #inst_id = cur_inst_id
    #vis_instance = test_instances[inst_id]
    # vis_instance = [0.61111111, 0.5,        0.69491525, 0.49166667]
    #vis_label = test_label[inst_id]
    width = cur_width

    # vis_dims = [1, 2, 3, 0]  # Dimensions to be analyzed
    # vis_dims = [0,1,2,4,5]
    #vis_dims = [i for i in range(len(vis_instance))]
    #if requested_dims is not None:
    #    vis_dims = requested_dims

    width_vec = np.array([width] * len(test_instances[0]))
    #res_vis_inst = np.array(vis_instance)[vis_dims]

    # set whole cert_vec = None only if all dimensions are analyzed
    # use the definition above for subspaces
    #cert_vec = None
    # cert_vec = [None if i in vis_dims else val for i, val in enumerate(vis_instance)]
    #if requested_dims is not None:
    #    cert_vec = [None if i in vis_dims else val for i, val in enumerate(vis_instance)]
    # cert_vec = [None if i in vis_dims else val for i, val in enumerate(vis_instance)]
    max_cube_data = vis_instance + np.array(width_vec) / 2
    min_cube_data = vis_instance - np.array(width_vec) / 2
    vis_inst_data = np.array([max_cube_data, min_cube_data])




    #vis_instance = test_instances[inst_id]
    #if change_feature_idx is not None:
    #    vis_instance[change_feature_idx] = vis_instance[change_feature_idx] * changed_feature_value

    #if changed_instance is not None:
    #    vis_instance = changed_instance

    near_train_insts = near_train_instances(train_data, y_train_data, [vis_instance], radius=width_vec / 2)


    idcs = list(range(len(vis_instance)))
    all_vis_dims = list(combinations(idcs, 2))

    tempBorders = {}
    dataframe_dict = {}
    for vis_dims in all_vis_dims:
        vis_dims = list(vis_dims)

        res_vis_inst = np.array(vis_instance)[vis_dims]
        cert_vec = [None if i in vis_dims else val for i, val in enumerate(vis_instance)]

        max_cube_data = vis_instance + np.array(width) / 2
        min_cube_data = vis_instance - np.array(width) / 2
        vis_inst_data = np.array([max_cube_data, min_cube_data])
        #vis_label = test_label[inst_id]

        vis_dbs = DecisionBoundaries(model=current_model, data=vis_inst_data, labels=vis_label)
        vis_dbs.getLinearRegions(certainty_vector=cert_vec, version=3)
        vis_dbs.getDecisionBoundariesExact(certainty_vector=cert_vec)



        if cert_vec is None:
            distances = vis_dbs.getDistancestoallBoundariesOpt(vis_instance)
        else:
            distances = vis_dbs.getDistancestoallBoundariesOpt(res_vis_inst)

        if len(distances[0]) > 0:
            arg_min = np.argmin(distances[0])
            min_dist = distances[0][arg_min]
            nearest_db_pt = distances[1][arg_min]

            df = pd.DataFrame()
            df['Instance'] = res_vis_inst
            df['Nearest DB point'] = nearest_db_pt
            db_dir = nearest_db_pt - res_vis_inst
            df['Direction to nearest DB'] = db_dir
            df['overall distance'] = min_dist

            df = df.transpose()
        else:
            df = pd.DataFrame()
            df['Instance'] = res_vis_inst
            df['Nearest DB point'] = None
            # db_dir = nearest_db_pt - res_vis_inst
            df['Direction to nearest DB'] = None
            df['overall distance'] = None

            df = df.transpose()
        dataframe_dict[tuple(vis_dims)] = df
        tempBorders[tuple(vis_dims)] = vis_dbs.exact_decision_boundaries

    return [vis_instance, vis_inst_data, near_train_insts, tempBorders, dataframe_dict]

'''


def createGrid(instance, varibleFeatures, lowerEnds, upperEnds, stepSize, loaded_model):
    grid_points = []
    for i in range(len(instance)):
        if i in varibleFeatures:
            values = np.linspace(lowerEnds[i], upperEnds[i],
                                 int((upperEnds[i] - lowerEnds[i]) // stepSize) + 1).tolist()
            grid_points.append(values)
        else:
            grid_points.append([instance[i]])

    all_combinations = []
    for combination in np.array(np.meshgrid(*grid_points)).T.reshape(-1, len(instance)):
        all_combinations.append(combination.tolist())

    probs = model_wrapper(all_combinations, loaded_model)
    all_logits = np.array(probs)

    probabilities = np.exp(all_logits) / np.sum(np.exp(all_logits), axis=1, keepdims=True)
    max_values = [[np.max(sublist)] for sublist in probabilities]
    return all_combinations, max_values, probabilities




PLATZHALTER = 'iris'
#PLATZHALTER = 'fetal_health'
model = load_model_from_updated_path(PLATZHALTER)

if __name__ == "__main__" and 1:
    print('Test')