# -*- coding: utf-8 -*-

import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cdd
import torch
import time
import cProfile
from polytope_package_v4 import CPU
from LinearSegmentsTest_v3 import Model, Cube, get_restricted_cpu, loadModel, getLinearSegments
from distribution_v3 import Distribution, propagate_adf, global_method, make_histo_grid
from scipy.spatial import ConvexHull, QhullError
from scipy.optimize import linprog, minimize

# Finds the closest point to x in the convex hull induced by the points in vertices.
# Does so via an optimization process over the weighted sum of the vertices.
def closest_point_in_hull_optimization(x, vertices):
    
    def objective_function(weights, x, vertices):
        point = np.zeros(shape=np.shape(vertices[0]))
        for i, vert in enumerate(vertices):
            point += vert*weights[i]
        return np.linalg.norm(point - x)

    def convex_combination_constraint(point):
        return np.sum(point) - 1  # Ensure convex combination: sum of coefficients = 1
    
    initial_guess = np.ones(len(vertices)) / len(vertices)
    initial_guesses = np.random.dirichlet(alpha = 1*np.ones(len(vertices)), size = 25)
    # initial_guess = np.zeros(len(vertices))# / len(vertices)     
    # initial_guess[0] = 1
    bounds = [(0, 1) for _ in range(len(vertices))] 
    constraints = [{'type': 'eq', 'fun': convex_combination_constraint}]
    curr_min_dist = 1
    for guess in initial_guesses:
        result = minimize(objective_function, guess, args=(x,vertices,), bounds=bounds, constraints=constraints, tol = 0.00001, options={"maxiter":10000, "disp":False})
        closest_point = np.dot(result.x, vertices)
        dis_to_x = np.linalg.norm(closest_point - x)
        if dis_to_x < curr_min_dist:
            curr_min_dist = dis_to_x
            closest_pt_overall = closest_point
        
    return closest_pt_overall


def are_points_on_hyperplane(points):
    # Extract coordinates of the points
    coordinates = np.array(points)
    
    # Create the coefficient matrix A and constant vector b
    A = coordinates[:, :-1]  # Exclude the last column (constant term)
    b = coordinates[:, -1]   # Last column represents the constant term
    
    # Use least squares to solve the linear system
    _, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    # If residuals are very small (close to zero), the points lie on the hyperplane
    return np.allclose(residuals, 0)


# Technical helper function dealing with the many high dimensional convex hull
# problems in the best ways possible.
def regularizeCH(pts):
    try:
        ch = ConvexHull(pts)
        vol = ch.volume
        ch_success = True
        return vol, ch_success
    except QhullError as qe:
        ch_success = False
        error_code = str(qe)[2:6]
        if error_code == '6154':
            # print("flat simplex")
            # print("Points on hyperplane? ", are_points_on_hyperplane(pts))
            if are_points_on_hyperplane(pts) == True:
                vol = 0
            else:
                ch = ConvexHull(pts, qhull_options = "QJ")
                vol = ch.volume
                return vol, ch_success
            
        elif error_code == '6271':
            print("topological error")
            ch = ConvexHull(pts, qhull_options = "QJ")
            vol = ch.volume
            ch_success = True
        
        elif error_code == '6347':
            print("precision error")
            ch = ConvexHull(pts, qhull_options = "QJ")
            vol = ch.volume
            ch_success = True
        
        elif error_code == '6214':
            print("not enough vertices for polytope")
            vol = 0
        
        elif error_code == '7088':
            try:
                ch = ConvexHull(pts, qhull_options = "Pp")
            except:
                ch = ConvexHull(pts, qhull_options = "QJ")
            vol = ch.volume
            ch_success = True
            
        else:
            print("not addressed error code: ", str(qe)[2:40], "...")
            try:
                ch = ConvexHull(pts, qhull_options = "QJ")
                vol = ch.volume
            except:
                vol = 0
            # ch.success = True
        return vol, ch_success
    



def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def softmax2(x):
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=0)

def getTorchModel(model_path):
    param_dict = {}
    model = torch.load(model_path)
    model.eval()
    for i, param in enumerate(model.parameters()):
        if i%2 == 0:
            param_dict["weights_" + str(int(i/2))] = param.detach()
        else:
            param_dict["bias_" + str(int((i-1)/2))] = param.detach()

    torch.save(param_dict, "synth/model.pt")

def regularizeCDDMat(cdd_mat, reg_type = 1, nr_eq = 1, print_message = False):
    reg_count = 0
    if reg_type == 1:
        
        poly_flag = True
        while poly_flag == True:
            try:     
                poly = cdd.Polyhedron(cdd_mat)
                poly_flag = False
            except:
                if print_message:
                    print("except: regularize cdd matrix type 1, reg count is ", reg_count)
                np_mat = np.array(cdd_mat)
                np_mat_1 = np_mat[:,0]  
                np_mat_2 = np_mat[:,1:]
                # Regularisierung
                np_mat_2 += ((-1)**reg_count)*(10**(-8 + reg_count))*np.ones(np_mat_2.shape)
                np_mat_2 = np.column_stack((np_mat_1, np_mat_2))
                
                cdd_mat = cdd.Matrix(np_mat_2, number_type = 'float')
                cdd_mat.rep_type = cdd.RepType.GENERATOR
            reg_count += 1
                
        return poly
    
    elif reg_type == 2:
        #only use for regularizations of inequality matrices
        poly_flag = True
        while poly_flag == True:
            try:
                poly = cdd.Polyhedron(cdd_mat)
                poly_flag = False
            except:
                if print_message:
                    print("except: regularize cdd matrix type 2, reg count is ", reg_count)
                np_mat = np.array(cdd_mat)
                # np_mat_1 = np_mat[:-nr_eq]  
                # np_mat_2 = np_mat[-nr_eq:]
                # np_mat *= 1.1
                np_mat *= 1.00001
                np_mat += ((-1)**reg_count)*(10**(-8 + reg_count))*np.ones(np_mat.shape)
                
                cdd_mat = cdd.Matrix(np_mat, number_type = 'float')
                # cdd_mat.extend(np_mat_2)
                cdd_mat.rep_type = cdd.RepType.INEQUALITY
            reg_count += 1

        return poly

# Computes the distance of a point to a hyperplane induces by hyperplane_points.
def distance_to_hyperplane(hyperplane_points, point):
    # Convert input arrays to numpy arrays for ease of computation
    hyperplane_points = np.array(hyperplane_points)
    point = np.array(point)
    
    
    # Ensure hyperplane_points has at least two points
    if len(hyperplane_points) < len(hyperplane_points[0]):
        # print(hyperplane_points)
        # raise ValueError("Need more points to define proper hyperplane.")
        return 1000, 1000*hyperplane_points[0], 1000*hyperplane_points[0]
        
    else:
        # Calculate the normal vector of the hyperplane and the projection to this plane
        v = hyperplane_points[1:] - hyperplane_points[0]
        normal_vector = np.linalg.svd(v)[2][-1]
        proj_onto_hyperplane= point - np.dot(point - hyperplane_points[0], normal_vector) * normal_vector
        # Calculate the distance from the point to the hyperplane
        distance = np.abs(np.dot(point - hyperplane_points[0], normal_vector))
    
        return distance, proj_onto_hyperplane, normal_vector

# Checks if x is in the convex hull induced by points. Does so utilizing 
# linear programming.
def in_hull(points, x):
    n_points = len(points)
    c = np.zeros(n_points)
    A = np.r_[points.T,np.ones((1,n_points))]
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success

def checkOverlap(polytope, cube, poly_pt_list):
    # get cube inequalities
    cb_vertices = [np.append(np.ones(1), cb_pt) for cb_pt in cube.points]
    mat_cb = cdd.Matrix(cb_vertices, number_type='float')
    mat_cb.rep_type = cdd.RepType.GENERATOR
    poly = regularizeCDDMat(mat_cb, reg_type = 1)
    ineqs_cb = poly.get_inequalities()
    
    poly_vertices = [np.append(np.ones(1), poly_pt_list[pt_idx]) for pt_idx in polytope.points]
    mat_poly = cdd.Matrix(poly_vertices, number_type='float')
    mat_poly.rep_type = cdd.RepType.GENERATOR
    poly = regularizeCDDMat(mat_poly, reg_type = 1)
    ineqs_poly = poly.get_inequalities()
    
    mat = cdd.Matrix(ineqs_cb)
    mat.extend(ineqs_poly)
    mat.rep_type = cdd.RepType.INEQUALITY
    poly = regularizeCDDMat(mat, reg_type = 2)
    gens = np.array(poly.get_generators())
    
    # if len(gens.shape) == 0:
    #     return False, 0
    if (gens.shape[0] == 0):
        return False, 0, None
    else:
        try:
            gens = gens[:,1:]
            ch = ConvexHull(gens)
            vol = ch.volume
    
            return True, vol, gens
        except:
            print("degenerate overlap with polytope")
            gens = gens[:,1:]
            return False, 0, gens

def propagateLinReg(points, cpu, label ="average"):

    output_points = []
    for point in points:
        dimension = len(point)
        errors = []
        for i in range(0, len(cpu.polytopes)):
            eq= cpu.polytopes[i].equations
            pol_errors = []
            for j in range(0, len(eq)):
                #print(point, eq[j][0:dimension], eq[j][dimension])
                #if np.dot(point, eq[j][0:dimension]) + eq[j][dimension] > 0:
                if np.dot(point, eq[j][0:dimension]) + eq[j][dimension] > 0:
                    pol_errors.append(np.dot(point, eq[j][0:dimension]) + eq[j][dimension])
                else:
                    pol_errors.append(0.0)
            if label=='average':
                errors.append(np.mean(pol_errors))
            if label== 'max':
                errors.append(np.max(pol_errors))
       
        index = [i for i in range(0, len(errors)) if errors[i]==min(errors)][0]
        point = np.matmul(cpu.polytopes[index].aff_lin[0], point) + cpu.polytopes[index].aff_lin[1]
        output_points.append(point)
   
    return output_points

def analyseInstances(met, test_labels):
    dicts = []
    for idx, inst in enumerate(met[0]):
        inst_dict = { "trivial": 0, "SM Values" : 0, "correct model pred" : 0, "SM Av" : 0, "SM Min" : 0, "SM Max" : 0}
        print(met[5][idx][0])
        if met[5][idx][0] == True:
            inst_dict["trivial"] = True
        else:
            inst_dict["trivial"] = False
        inst_dict["SM Values"] = met[5][idx][1]
        inst_dict["SM Av"] = sum(met[5][idx][1])/len(met[5][idx][1])
        inst_dict["SM Min"] = min([max(x) for x in met[5][idx][1]])
        inst_dict["SM Max"] = max([max(x) for x in met[5][idx][1]])
        if test_labels[idx] == met[6][idx]:
            inst_dict["correct model pred"] = True
        else:
            inst_dict["correct model pred"] = False
        
        dicts.append(inst_dict)
    return dicts

def near_train_instances(data_train, y_train, instances, radius = 0.1, mode = "cube"):
    for instance in instances:
        instances_in_cube = []
        classes_in_cube = []    
        dim = len(data_train[0])
        cube_lengths = []
        for i in range(dim):
            min_i = min([y[i] for y in data_train])
            max_i = max([y[i] for y in data_train])
            cube_lengths.append(radius[i] * (max_i-min_i))
        for j in range(len(data_train)):
            in_cube = True
            if mode == "cube":
                for i in range(dim):
                    if abs(instance[i]-data_train[j][i]) > cube_lengths[i]:
                        in_cube = False
            if mode == "circular":
                sum_of_components = 0   
                for i in range(dim):
                    sum_of_components += (data_train[j][i] - instance[i])**2/(cube_lengths[i]**2)
                if sum_of_components > 1:
                    in_cube = False
            if in_cube:
                instances_in_cube.append(data_train[j])
                classes_in_cube.append(y_train[j])
    return instances_in_cube, classes_in_cube

def plot_2d_criteria(test_criteria, crits_to_plot, test_labels, test_preds, pvals, crit1_step, crit2_step, mode = "both"):
                      # inst_criteria, test_inst_criteria, preds, test_labels, test_preds, instances = None, test_instances = None,
                      #                       max_met_diff = 0.05, max_distance = None, test_alpha = 0.05, test_p = 0.5):
    crit1 = crits_to_plot[0]
    crit2 = crits_to_plot[1]
    crit_vals_plot = test_criteria[:, [crit1, crit2]]
    
    criteria_list = ["metric 1 class 0", "metric 1 class 1", "distance to decision boundary", "true label class 0, correct model prediction",
                     "true label class 0, wrong model prediction", "true label class 1, wrong model prediction", "true label class 1, correct model prediction"]
    
    # assert instances to their bins
    min_vals = np.min(crit_vals_plot, axis = 0)
    max_vals = np.max(crit_vals_plot, axis = 0)
    bin_width_0 = (max_vals[0] + 0.001 - min_vals[0]) / crit1_step
    bin_width_1 = (max_vals[1] + 0.001- min_vals[1]) / crit2_step
    bin_centers_0 = np.linspace(min_vals[0], max_vals[0], num = crit1_step)
    bin_centers_1 = np.linspace(min_vals[1], max_vals[1], num = crit2_step)
    
    bins_0 = np.floor((crit_vals_plot[:,0] - min_vals[0]) / bin_width_0)
    bins_1 = np.floor((crit_vals_plot[:,1] - min_vals[1]) / bin_width_1)
    inst_bins = np.column_stack((bins_0, bins_1))
    
    if mode == "both":
        n_map = np.zeros((crit1_step, crit2_step))
        k_map = np.zeros((crit1_step, crit2_step))
        pval_map = np.zeros((crit1_step, crit2_step))  
        
        for i, inst in enumerate(test_criteria):
            inst_bin = inst_bins[i]
            n_map[int(inst_bin[0]), int(inst_bin[1])] += 1
            pval_map[int(inst_bin[0]), int(inst_bin[1])] += pvals[i]
            if test_labels[i] == test_preds[i]:
                k_map[int(inst_bin[0]), int(inst_bin[1])] += 1
        
        acc_map = k_map / n_map
        pval_map = pval_map / n_map
        # acc_map = np.nan_to_num(acc_map)
    
    # plot accuracy wrt criteria
    colors = [(0, 'red'), (0.8, 'orange'), (1, 'green')]
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
    
    plt.imshow(acc_map, cmap = cmap)
    plt.yticks(np.arange(len(bin_centers_0)), np.round(bin_centers_0, decimals=2))
    plt.xticks(rotation = 90)
    plt.xticks(np.arange(len(bin_centers_1)), np.round(bin_centers_1, decimals=2))
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.title(f'accuracy of the criteria {criteria_list[crit1]} and {criteria_list[crit2]}')
    plt.ylabel(f'Axis {criteria_list[crit1]}')
    plt.xlabel(f'Axis {criteria_list[crit2]}')
    plt.show()
    
    # plot number of instances wrt criteria
    colors1 = [(0, 'white'), (0.1, 'orange'), (1, 'green')]
    cmap1 = LinearSegmentedColormap.from_list('custom_cmap', colors1)
    
    plt.imshow(n_map, cmap = cmap1)
    plt.yticks(np.arange(len(bin_centers_0)), np.round(bin_centers_0, decimals=2))
    plt.xticks(rotation = 90)
    plt.xticks(np.arange(len(bin_centers_1)), np.round(bin_centers_1, decimals=2))
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.title('Number of Instances')
    plt.ylabel(f'Axis {criteria_list[crit1]}')
    plt.xlabel(f'Axis {criteria_list[crit2]}')
    plt.show()
    
    # plot p value of instances wrt criteria
    colors2 = [(0, 'green'), (0.1, 'orange'), (1, 'red')]
    cmap2 = LinearSegmentedColormap.from_list('custom_cmap', colors2)
    
    plt.imshow(pval_map, cmap = cmap2)
    plt.yticks(np.arange(len(bin_centers_0)), np.round(bin_centers_0, decimals=2))
    plt.xticks(rotation = 90)
    plt.xticks(np.arange(len(bin_centers_1)), np.round(bin_centers_1, decimals=2))
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.title('P Values')
    plt.ylabel(f'Axis {criteria_list[crit1]}')
    plt.xlabel(f'Axis {criteria_list[crit2]}')
    plt.show()
        

    return inst_bins
    
    
    
    
    
    
    
    