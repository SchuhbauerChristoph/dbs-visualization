# -*- coding: utf-8 -*-

import copy
import numpy as np
from scipy.spatial import Delaunay
import cdd 
import pickle
from scipy.linalg import null_space, orth
from scipy.spatial import ConvexHull
from itertools import chain


def get_indices_of_array(base_list, test_array):
    
    if type(base_list) == list:
        test_np = np.array(base_list)
    else:
        test_np = base_list
    index = np.where(np.all(test_np == test_array, axis=1))[0]
    if len(index) > 0:
        return [index[0]]
    else:
        return []

    if False:
        if np.any(np.all(test_array == base_list, axis=1)):
            test_np = np.array(base_list)
            idx = np.where(np.all(test_np == test_array, axis=1))[0][0]
            #print("case in list")
            index_list = [idx]
            return index_list
        else:
            return []

def max_diff(array_list):
    dist_matrix = np.zeros((len(array_list), len(array_list)))
    for i in range(len(array_list)):
        for j in range(len(array_list)):
            dist_matrix[i][j] = np.linalg.norm(array_list[i]-array_list[j])
    return np.max(dist_matrix)      
    

def get_colorschemes(k, dim):
    lst=[([0],[])]
    while len(lst[0][0])<dim*k:
        templst=[]
        for (scheme,jumps) in lst:
            templst.append((scheme+[scheme[-1]],jumps))
            if scheme[-1] != dim-1:
                if len(scheme)%dim != 0:
                    templst.append((scheme+[scheme[-1]+1],jumps+[len(scheme)%dim]))
                else:
                    templst.append((scheme+[scheme[-1]+1],jumps))
        lst=templst
    templst=copy.deepcopy(lst)
    for (scheme,jumps) in lst:
        if len(set(jumps))!=dim-1:
            templst.remove((scheme,jumps))
    lst=templst
    lst_of_matrices=[]
    for (scheme,jumps) in lst:
        matrix=[]
        for row in range(0,k):
            temp=[]
            for column in range(0,dim):
                temp.append(scheme[row*dim+column])
            matrix.append(temp)
        matrix=np.array(matrix)
        lst_of_matrices.append(matrix)
    return lst_of_matrices


def colorscheme_to_simplex(matrix,simplex,index_to_pt_list,known_combs):
    dimension=len(index_to_pt_list[0])
    aktuelle_itpl=[index_to_pt_list[i] for i in simplex.points]
    points=[]
    conns=[]
    for i in range(0,len(matrix[0])):
        column=matrix[:,i]
        if tuple(column) in known_combs.keys():
            point=known_combs[tuple(column)]
            points.append(point)
        else:
            point=np.zeros(dimension)
            for x in column:
                point+=aktuelle_itpl[x]
            point=point*(1/len(column))    
            index_to_pt_list.append(point)
            points.append(len(index_to_pt_list)-1)
            known_combs[tuple(column)]=len(index_to_pt_list)-1
    for j in range(0,len(points)):
        for l in range(0,len(points)):
            if j>l:
                conns.append((l,j))
    simplex=Polytope(points,conns,simplex.aff_lin)
    return simplex

def prep_for_eval_on_a_point(LinearSegment):
    W, b = LinearSegment.aff_lin
    N = null_space(W)
    R = orth(W)
    if N.size == 0:
        if R.shape[0] == R.shape[1]:
            #bijective 
            W_inv = np.linalg.inv(W)
            det_W_inv = np.abs(np.linalg.det(W_inv))
            return (0, W_inv, det_W_inv)
        else:
            #injective, but not surjective
            RT = np.transpose(R)
            W_tilde = RT@W
            b_tilde = RT@b
            W_tilde_inv = np.linalg.inv(W_tilde)
            det_W_tilde_inv = np.abs(np.linalg.det(W_tilde_inv))
            return (1, W_tilde_inv, det_W_tilde_inv, RT, b_tilde)

    else:
        if R.shape[1] == W.shape[0]:
            #non injective, but surjective
            
            W_pi = np.linalg.pinv(W)
            N_orth = orth(W_pi)
            
            W_tilde = W@N_orth
            W_tilde_inv = np.linalg.inv(W_tilde)
            det_W_tilde_inv = np.abs(np.linalg.det(W_tilde_inv))
            
            return (2, det_W_tilde_inv)
        else:
            #non injective, non surjective
            RT = np.transpose(R)
            
            W_pi = np.linalg.pinv(W)
            N_orth = orth(W_pi)
            
            W_tilde = RT@W@N_orth
            W_tilde_inv = np.linalg.inv(W_tilde)
            det_W_tilde_inv = np.abs(np.linalg.det(W_tilde_inv))
            return (3, det_W_tilde_inv) 


# Klasse eines Polytopes.
# Attribute:
#   points: Menge von Indizes der Eckpunkte des Polytops
#   conn: Menge von Paaren von Indizes von Punkten, aufsteigend geordnet; Diese
#         Menge muss alle Randkanten enthalten
#   aff_lin: affin lineare Abbildung geltend auf dem Polytop
# Vorschlaege: 
#         Hinzufuegen von current_point_union und input_point_union (siehe CPU)
#         als Attribute. Ist keine unnoetige Speicherverschwendung (im Gegensatz
#         dazu, was ich bei der letzten Besprechung meinte), da Listen in
#         Python immer als Referenzen gehaendelt werden.
#         Ich denke aber, dass es wenn dann sinvoll wäre diese Attribute niemals
#         durch eine Methode der Klasse Polytope zu modifizieren
class Polytope():
    def __init__(self, points, conn, aff_lin):
        self.points = list(points)
        self.conn = conn
        self.aff_lin = aff_lin
        self.equations = None
        self.output_equations = None 
    
    #Zerlegt das Polytop in Simplizes nach Delaunay
    #Wird aufgerufen in "CPU.subdivide".
    def delaunay_triang(self, index_to_pt_list):
        simps = []
        pt_list = [index_to_pt_list[i] for i in self.points]
        if len(index_to_pt_list[0]) > 1:
            tri = Delaunay(pt_list)        
            for sim in tri.simplices:
                sim_pts = [self.points[j] for j in sim]
                sim_conns = []
                for i in sim_pts:
                    for j in sim_pts:
                        if i>j:
                            sim_conns.append((j,i))
                simps.append(Polytope(sim_pts, sim_conns, self.aff_lin))
            return simps
        else:
            return [self]

    #Zerlegt einen Simplex in kleiner Simplizes mit edgewise subdivision zum Parameter k.
    #Wird aufgerufen in "CPU.subdivide".
    def edgewise(self, index_to_pt_list, k = 2, known_colorschemes={}):          
       dim=len(self.points)
       known_combs1={}
       
       if (dim,k) in known_colorschemes.keys():
           colorschemes=known_colorschemes[(dim,k)]
       else:
           colorschemes=get_colorschemes(k, dim)
           known_colorschemes[(dim,k)]=colorschemes
       simplices=[]
       for matrix in colorschemes:
           simplices.append(colorscheme_to_simplex(matrix, self, index_to_pt_list, known_combs1))
       return simplices       
            
   
    
    
# Klasse konvexes Polytope Union.
# Attribute:
#   input_points_union: Liste aller Punkte, die in der Polytopezerlegung des
#   Inputs vorkommen
#   current_points_union: Liste aller Punkte der Polytopzerlegung, propagiert
#   in die aktuell betrachtete Zwischenschicht. Es gilt current_points_union[k]
#   ist das Propagationsergebnis von input_points_union[k]
#   conn: Menge aller Verbindungen der Polytopzerlegung
#   polytopes: Menge aller Polytope der Polytopzerlegung
class CPU():
    
# Initialisiert eine cpu bestehend aus einem Polytop von der Menge von vertices
# des Polytops
    def __init__(self, input_points, conn):
        self.input_points_union = copy.deepcopy(input_points)
        self.input_points_array = np.array(self.input_points_union)
        
        # initialisiere die lineare Abbildung als Tupel von Identität und null-bias
        aff_lin = (np.identity(input_points[0].size),
                   np.zeros(input_points[0].size))
        
        self.polytopes = list(set([Polytope(set(range(len(input_points))), conn, aff_lin)]))
        
        self.current_points_union = copy.deepcopy(input_points)
        self.conn = conn
        self.subdivision = None
        self.lin_seg_preps = None
    
    def linear_transform(self, weights = None, bias = None):
        # nehme hier weights als numpy matrix und bias optional als numpy array
        # initialisiere leeren Bias in der richtigen Dimension  und Identität
        if weights is None:
            weights = np.identity(self.point_dim)
            
        if bias is None:
            bias = np.zeros(weights.shape[1])
            
        for i, point in enumerate(self.current_points_union):
            self.current_points_union[i] = np.matmul(weights, point) + bias
        
        for P in self.polytopes:
            P.aff_lin = (
                weights @ P.aff_lin[0], weights @ P.aff_lin[1] + bias
                )
            
        
    # Wende LeakyReLU entlang der Dimension k an
    # alpha ist der LeakyReLU Parameter, bei uns normalerweise 0.01.
    
    
    def divide_by_activation(self, dimension, alpha = 0.01):
        
        # Korrigiere Dimension für np.array
        dim = dimension -1
        
        # Liste zum Kodieren, welche Punkte ueber und unter der Hyperebene 
        # {x_dim = 0} liegen.
        point_upper_lower_list = [0] * len(self.current_points_union)
        
        # Listenreihenfolge ist gleich der von self.input_points_union und
        # self.current_points_union
        for k, point in enumerate(self.current_points_union):
            if point[dim] > 0:
                point_upper_lower_list[k] = 1
            elif point[dim] < 0:
                point_upper_lower_list[k] = -1
            else:
                point_upper_lower_list[k] = 0
        
        # Menge aller Kanten, mit einem Punkt oberhalb und einem Punkt unter-
        # halb der Hyperebene
        conn_intersec = set(
            (a,b) for (a,b) in self.conn if 
            np.abs(point_upper_lower_list[a]-point_upper_lower_list[b]) == 2
            )
        
        # Dictionary, das für jede Kante in conn_intersec die Strecken
        # kodiert, in die sich die Kante aufspaltet. 
        # Es gilt conn_decomp[P] = (A,B,intersec_idx) für A die 
        # Strecke unterhalb der Hyperebene, B die Strecke oberhalb der
        # Hyperebene und intersec_idx den Index des Schnitts mit der Hyperebene
        # in self.current_points_union.
        conn_decomp = dict.fromkeys(conn_intersec)
        for (a,b) in conn_intersec:
            # Berechnung des Schnitts
            curr_point1 = self.current_points_union[a]
            curr_point2 = self.current_points_union[b]
            curr_direc = curr_point2 - curr_point1
            
            input_point1 = self.input_points_union[a]
            input_point2 = self.input_points_union[b]
            input_direc = input_point2 - input_point1
            
            s = -curr_point1[dim]/curr_direc[dim]
            input_intersec = input_point1 + s * input_direc
            curr_intersec = curr_point1 + s * curr_direc
            
            # Ggf. hinzufuegen von input_intersec zu input_points_union
            idx = get_indices_of_array(self.input_points_array, input_intersec)
            # idx = get_indices_of_array(self.input_points_union, input_intersec)
            if idx == []:
                self.input_points_union.append(input_intersec)
                self.input_points_array = np.append(self.input_points_array, [input_intersec], axis = 0)
                self.current_points_union.append(curr_intersec)
               
                
                input_intersec_idx = len(self.input_points_union)-1
                
                # Updaten von point_upper_lower_list
                point_upper_lower_list.append(0)
                
            else:
                input_intersec_idx = idx[0]
                
                # Kompatibilitaet mit point_upper_lower_list_pruefen
                if point_upper_lower_list[input_intersec_idx] != 0:
                    print('Achtung: Kompatibilitaet point_upper_lower_list')
            
            # Erstellen der neuen Kanten; beginnend mit der Unteren
            if point_upper_lower_list[a] < point_upper_lower_list[b]:
                conn_lower = order((a, input_intersec_idx))
                conn_upper = order((input_intersec_idx, b))
            else:
                conn_lower = order((b, input_intersec_idx))
                conn_upper = order((input_intersec_idx, a))
            
           
            conn_decomp[(a,b)]= (conn_lower, conn_upper, input_intersec_idx)
        
        # Update der Menge von Polytopen
        new_polytopes = set()
        for P in self.polytopes:
            P.points = set(P.points)
            # P_lower_points und P_upper_points werden die Vertices des
            # Polytops unterhalb und ueberhalb der Hyperebene kodieren.
            P_lower_points = set(k for k in P.points if point_upper_lower_list[k] <= 0)
            P_upper_points = set(k for k in P.points if point_upper_lower_list[k] >= 0)

            #P_upper_points_new = P_upper_points.copy()
            
            # abhandeln des Falls, dass P komplett ueber der Hyperebene liegt
            if P_upper_points ==  P.points:
                new_polytopes.add(P)
                
            # abhandeln des Falls, dass P komplett unterhalb der Hyperebene liegt
            elif P_lower_points == P.points:
                # updaten der affin linearen Abbildung; koennte zu Methode von
                # Polytope gemacht werden
                shape = P.aff_lin[1].shape
                x = np.ones(shape)
                x[dim] = alpha
                P.aff_lin = (np.diag(x)@P.aff_lin[0], np.diag(x)@P.aff_lin[1])
                
                new_polytopes.add(P)
            
            # betrachten des interessanten Falls: Das Polytop hat Werte unter-
            # halb und ueberhalb der Hyperebene
            else:
              
                # Menge der Schnittpunkte von P mit der Hyperebene, muessen
                # noch zu P_lower_points und P_upper_points hinzugefuegt wer-
                # den
                
             
                P_new_points = {conn_decomp[(a,b)][2] for (a,b) in P.conn 
                              if (a,b) in conn_intersec}
                
             
                #P_lower_points.update(P_new_points)
                
                # Erstellen der nicht reduzierten Menge Kanten des unteren Poly-
                # tops
                # Alle Kanten von P, die unterhalb der Hyperebene liegen
                P_lower_conn = set((a,b) for (a,b) in P.conn if point_upper_lower_list[a]<=0
                                   and point_upper_lower_list[b]<=0)
                
                
                
                # Fuer jede Kante von P, die die Hyperebene schneidet, hinzufue-
                # gen der Kante, die unterhalb der Hyperebene liegt
                P_lower_conn.update([conn_decomp[(a,b)][0] for (a,b) in P.conn
                                    if (a,b) in conn_intersec])
                
                # Fuer je zwei Punkte auf der Hyperebene eine Kante hinzufuegen
                P_points_on_hp = P_new_points.union(
                    {k for k in P.points if point_upper_lower_list[k] == 0}
                    )
                
                P_new_conn = []
                P_points_on_hp_extended = list(P_points_on_hp)
                if P_upper_points != set([]):
                    P_points_on_hp_extended.append(list(P_upper_points)[0])
                else:
                    print("upper points empty")
                try:
                    c_hull = ConvexHull([self.input_points_union[temp_idx] for temp_idx in P_points_on_hp_extended])
                except:
                    c_hull = ConvexHull([self.input_points_union[temp_idx] for temp_idx in P_points_on_hp_extended], qhull_options="QJ")
                hull_verts = [P_points_on_hp_extended[idx] for idx in c_hull.vertices[:-1]]
                hull_verts = set(hull_verts)
            
                # print("Hull verts: ", hull_verts)
                #red_idx_list = [l for (k,l) in enumerate(P_points_on_hp_extended) if k in hull.vertices]
                for simp_fac in c_hull.simplices:
                    for s_vert in simp_fac:
                        for s_vert_2 in simp_fac:
                            s_vert_idx = P_points_on_hp_extended[s_vert]
                            s_vert_idx_2 = P_points_on_hp_extended[s_vert_2]
                            if s_vert_idx < s_vert_idx_2 and s_vert != len(P_points_on_hp_extended)-1 and s_vert_2 != len(P_points_on_hp_extended)-1:
                               P_new_conn.append((s_vert_idx,s_vert_idx_2)) 
                P_new_conn = set(P_new_conn)
                
                # except:
                #     print("Reduction not possible")
                #     # continue
                
                P_lower_points.update(hull_verts)
                P_lower_conn.update(P_new_conn)
                
                # P_lower_conn_dict = preprocess_tuples(P_lower_conn)
                # reduzieren, der unteren Punkte Menge und der unteren Kanten-
                # menge
                
                P_lower_points_reduced = P_lower_points
                P_lower_conn_reduced = P_lower_conn
                
                # Updaten der affin linearen Abbildung
                shape = P.aff_lin[1].shape
                x = np.ones(shape)
                x[dim] = alpha
                P_lower_aff_lin = (np.diag(x)@P.aff_lin[0], np.diag(x)@P.aff_lin[1])
               
                # Hinzufuegen des unteren Polytops
              
                new_polytopes.add(Polytope(P_lower_points_reduced,
                                          P_lower_conn_reduced,
                                          P_lower_aff_lin))
                
                # Analoge Konstruktion des oberen Polytops
                
                #P_upper_points_new.update(hull_verts)
                P_upper_conn = set((a,b) for (a,b) in P.conn if point_upper_lower_list[a]>=0
                                   and point_upper_lower_list[b]>=0)
                P_upper_conn.update([conn_decomp[(a,b)][1] for (a,b) in P.conn if
                                 (a,b) in conn_intersec])

                P_upper_points.update(hull_verts)
                P_upper_conn.update(P_new_conn)
                
            
                #same idea here
                P_upper_points_reduced = P_upper_points
                P_upper_conn_reduced = P_upper_conn
                
               
                new_polytopes.add(Polytope(P_upper_points_reduced,
                                          P_upper_conn_reduced,
                                         P.aff_lin))
            P.points = list(P.points)
        self.polytopes = list(new_polytopes)
        
        
        # Update der current_points_union
        for k in range(len(self.current_points_union)):
            if point_upper_lower_list[k] == -1:
                self.current_points_union[k][dim] = alpha * self.current_points_union[k][dim]
        
        # Update der Liste der connections
        self.conn = set().union(*[P.conn for P in self.polytopes])


    #class function getInputEquations
    #adds ttribute equations to every polytope in the CPU
    #lower-dim. polytopes in the input space are deleted from self.polytopes
    def getInputEquations(self):
        
        del_indices = []
        for i in range(0, len(self.polytopes)):
            polytope_points=[self.input_points_union[j] for j in self.polytopes[i].points]
            if len(self.input_points_union[0]) >=2:
                try:
                    polytope_equations = ConvexHull(polytope_points).equations
                    self.polytopes[i].equations = polytope_equations 
                except:
                    del_indices.append(i)
            else:
                polytope_equations = []
                for pt in polytope_points:
                    polytope_equations.append(np.array([-1, pt[0]]))
                self.polytopes[i].equations = np.array(polytope_equations)
               
        
        self.polytopes = [self.polytopes[i] for i in range(0, len(self.polytopes)) if i not in del_indices]
        
        return 

    #fügt Attribut output_equations für jedes Polytop in CPU hinzu 
    #für niedrig-dimensionale Polytope wird cdd-Paket zur Bestimmung der Ungleichungen verwendet
    #ansonsten die ConvexHull Funktion von scipy 
    def getOutputEquations(self):
        
        for i in range(0, len(self.polytopes)):
            polytope_points = [self.current_points_union[i] for i in self.polytopes[i].points]
            try:
                polytope_equations = ConvexHull(polytope_points).equations
                self.polytopes[i].output_equations = polytope_equations
            except:
                
                polytope_vertices = [np.append(np.ones(1), self.current_points_union[j]) for j in self.polytopes[i].points]
                mat = cdd.Matrix(polytope_vertices, number_type='float')
                mat.rep_type = cdd.RepType.GENERATOR
                poly = cdd.Polyhedron(mat)
                inequalities = poly.get_inequalities()
                equality_indices = inequalities.lin_set
                
                a = np.array(inequalities)
                b = -np.append(a[:,1:], np.reshape(a[:,0], (a.shape[0],1)), axis=1)
                #actually do not need the reshaping if I just correctly turn the rows of a into inequalities that are checked 
                for index in equality_indices:
                    columns = b.shape[1]
                    b = np.append(b, np.reshape(-b[index,:], (1,columns)), axis=0)
                
                self.polytopes[i].output_equations = b 
                    
        return 

    #checke für jeden Punkt in points, in welchen Polytopen des Outputraums er liegt 
    #points: Liste von np.arrays
    #eps: float, der Fehler, der beim Erfüllen der Ungleichungen eines Polytops zugelassen wird 
    #returns: Liste von Listen: i-ter Eintrag der Liste ist eine Liste mit den Indizes aller
    #Output-Polytope, in welchem der i-te Punkt in points enthalten ist 
    def checkPolytopes(self, points, eps=0):
      
        points_polytope_indices = []
        for point in points:
            point_polytope_indices = []
            for i in range(0, len(self.polytopes)):
                if self.polytopes[i].output_equations is not None:
                    
                    eq = self.polytopes[i].output_equations
                      
                    for j in range(0, len(eq)):
                        if np.dot(point, eq[j][0:-1]) > -eq[j][-1]+eps:
                            #if one equation is not fulfilled up to eps, skip this polytope 
                            break
                        #if all equations have been tested and fulfilled, add polytope i to list
                        if j==len(eq)-1:
                            point_polytope_indices.append(i)
                
                #der else-Fall ist nicht mehr notwendig sein, wenn das Attribut output_equations
                #für alle Polytope vorhanden ist
                #falls dies für ein Polytope nicht der Fall ist, wird mit cdd_in_hull geprüft,
                #ob der Punkt in diesem Polytop liegt 
                else:
                    polytope_vertices = [ np.append(np.ones(1), self.current_points_union[j]) for j in self.polytopes[i].points]
                    if cdd_in_hull(np.array(polytope_vertices), point, eps):
                        point_polytope_indices.append(i)
    
            points_polytope_indices.append(point_polytope_indices)
                    
        return points_polytope_indices
    
    #speichere ein Objekt der Klasse CPU als Dictionary mit pickle 
    #kann dieses Objekt mit Funktion loadCPU wieder laden 
    #muss noch genau prüfen, dass gespeichertes und geladenes Objekt sicher identisch sind 
    def saveCPU(self, filename= "cpu.pkl"):
        
        cpu_dict={}
        cpu_dict['current_points_union'] = self.current_points_union
        cpu_dict['input_points_union'] = self.input_points_union
        cpu_dict['subdivision'] = self.subdivision
        cpu_dict['lin_seg_preps'] = self.lin_seg_preps
        for i in range(0, len(self.polytopes)):
            cpu_dict[i] = {'points': self.polytopes[i].points,
                           'conn': self.polytopes[i].conn,
                           'aff_lin': self.polytopes[i].aff_lin,
                           'equations': self.polytopes[i].equations,
                           'output_equations': self.polytopes[i].output_equations}
        
        with open(filename, 'wb') as f:
            pickle.dump(cpu_dict, f)
        
        return 
    
    def prep_lin_seg_for_eval(self):
        list_of_preparations=[]
        for segment in self.polytopes:
            list_of_preparations.append(prep_for_eval_on_a_point(segment))
        self.lin_seg_preps=list_of_preparations

#zerlegt die Polytope einer gegebenen CPU weiter in Simplizes (mit dalaunay) und diese jeweils weiter
#in kleinere Simplizes mit edgewise subdivision zum Parameter k. Speichert das Ergebnis in der self.subdivision.
    def subdivide(self, k = 2, epsilon = None, mode = "delaunay"):
        edgewise_itpl = copy.deepcopy(self.input_points_union)
        subsimplices = []
        known_colorschemes_1={}
        tri_list = []
        for i, poly in enumerate(self.polytopes):
            bary_list = []
            if i%10 == 0:
                print("Polytope number " + str(i))
                
            
            if mode == "delaunay":
                bary = poly.delaunay_triang(edgewise_itpl)
                bary_only_points=[x.points for x in bary]
                tri_list.append(bary_only_points)
                
            if epsilon is None:
                for bar in bary:
                    edge = bar.edgewise(edgewise_itpl, k, known_colorschemes=known_colorschemes_1)
                    edge_only_points = [x.points for x in edge]
                    bary_list.append(edge_only_points)
            else:
                k_ad_list = []
                for bar in bary:
                    points_in_bar=[edgewise_itpl[j] for j in bar.points]
                    widest_edge = max_diff(points_in_bar)
                    k_ad = int(np.ceil(widest_edge/epsilon))
                    k_ad_list.append(k_ad)
                    edge = bar.edgewise(edgewise_itpl, k_ad, known_colorschemes=known_colorschemes_1)
                    edge_only_points=[x.points for x in edge]
                    bary_list+=edge_only_points
            subsimplices.append(bary_list)
            
        self.subdivision = tri_list, subsimplices, edgewise_itpl          
            

#lädt eine CPU, welche mit saveCPU() als Dictionary abgespeichert wird
#Parameter: filename: Name des gespeicherten Pickle-Files
#returns: Objekt der Klasse CPU mit den im Dictionary gespeicherten Werten
#falls das File nicht existiert, wird CPU-Objekt ohne Punkte, Polytope und Conns zurückgegeben  
def loadCPU(filename):
        
    try:
        with open(filename, 'rb') as f:
            cpu_dict = pickle.load(f)
    except:
        print("File does not exist!")
        return CPU([], [])
        
    cpu = CPU(cpu_dict['input_points_union'], [])
    cpu.current_points_union = cpu_dict['current_points_union']
    try:
        cpu.lin_seg_preps=cpu_dict['lin_seg_preps']
    except:
        cpu.lin_seg_preps = None
    try:
        cpu.subdivision = cpu_dict['subdivision']
    except:
        cpu.subdivision = None
    cpu.polytopes = []
    polytope_number_list = [key for key in cpu_dict.keys() if type(key)==int]
    polytope_number_list.sort()
    for i in polytope_number_list:
        p = Polytope(cpu_dict[i]['points'], cpu_dict[i]['conn'], cpu_dict[i]['aff_lin'])
        p.equations = cpu_dict[i]['equations']
        p.output_equations = cpu_dict[i]['output_equations']
        cpu.polytopes.append(p)
    
    return cpu 
            

#checke mit cdd, ob ein Punkt x in dem Polytop, welches von den Punkten in points
#aufgespannt wird, enthalten ist
#eps: Toleranz bei Prüfung der Ungleichungen 
#Funktion wird nur verwendet, falls ein Polytop kein Attribut output_equations gespeichert hat 
def cdd_in_hull(points, x, eps=10**(-10)):
  
    mat = cdd.Matrix(points, number_type='float')
    mat.rep_type = cdd.RepType.GENERATOR
    poly = cdd.Polyhedron(mat)
    
    inequalities = poly.get_inequalities()
    equality_indices = inequalities.lin_set
    
    a = np.array(inequalities)
   
    rows = a.shape[0]
    for i in range(0, rows):
        if i not in equality_indices:
            #check for inequality according to the way inequalities are saved in a by cdd package
            if -np.dot(a[i,1:].flatten(), x) > a[i,0] + eps:
                return False 
    
        else:
            #if equality is not fulfilled, return False as well
            #if we never reach a return False statement, we return True as all equalities have been satisfied 
            if abs(-np.matmul(a[i,1:], x) - a[i,0]) > eps:
                return False 
        
    return True 

# Zwei Hilfsfunktionen um Connections rauszufinden

def preprocess_tuples(tuple_list):
    tuple_dict = {}
    for x, y in tuple_list:
        if x in tuple_dict:
            tuple_dict[x].add(y)
        else:
            tuple_dict[x] = {y}
        if y in tuple_dict:
            tuple_dict[y].add(x)
        else:
            tuple_dict[y] = {x}
    return tuple_dict

def filter_tuples(tuple_dict, number_set):
    result = []
    for num in number_set:
        #if num in tuple_dict:
        for neighbor in tuple_dict[num]:
            if neighbor in number_set and neighbor > num:
                result.append((num, neighbor))
    return result

# Input:
#   pts_list: Liste von Punkten
#   pts_idx: Menge von Punkten, kodiert als Menge von Indizes der Liste pts_list
#   conn: Menge von Kanten, kodiert als aufsteigendes Tupel von Indizes der Liste pts_list
#   
# Output:
#   pts_reduced: Die Menge von Eckpunkten (kodiert als Indizes von pts_list) 
#   des Polytops aufgespannt von den Punkten pts_list[pts_idx];
#   conn_reduced: Eine möglichst reduzierte Teilmenge von conn, die alle 
#   Eckkanten enthaelt
# def reduce(pts_list, pts_idx, conn):
#     try:
#         pts_idx_list = list(pts_idx)
#         pts = [pts_list[k] for k in pts_idx]
#         hull = ConvexHull(pts, qhull_options = "")
#         pts_reduced =  [l for (k,l) in enumerate(pts_idx_list) if k in hull.vertices]
    
#         # Erstellen einer edges_reduced_list[k] alle Kanten des k-ten Simplex ent-
#         # haelt
#         conn_reduced_list = [None] * len(hull.simplices)
#         for k,s in enumerate(hull.simplices):
#             s_idx_list = [pts_idx_list[k] for k in s]
#             conn_reduced_list[k] = set(filter_tuples(conn, s_idx_list))
            
    
    
#         conn_reduced = set().union(*conn_reduced_list)
#         return set(pts_reduced), conn_reduced
    
#     except:
#         print("Reduction not possible")
      
#         return pts_idx, conn
def reduce(pts_list, pts_idx, conn):
    try:
        pts_idx_list = list(pts_idx)
        pts = [pts_list[k] for k in pts_idx]
        hull = ConvexHull(pts, qhull_options = "")
        pts_reduced =  [l for (k,l) in enumerate(pts_idx_list) if k in hull.vertices]
        # hull_simplices = hull.simplices
        # # Erstellen einer edges_reduced_list[k] alle Kanten des k-ten Simplex ent-
        # # haelt
        # simplices_dict = {}
        # for k,s in enumerate(hull_simplices):
        #     s_idx_list = set([pts_idx_list[a_idx] for a_idx in s])
        #     for point in s_idx_list:
        #         if not point in simplices_dict:
        #             simplices_dict[point] = s_idx_list
        #         else: 
        #             simplices_dict[point]=simplices_dict[point].union(s_idx_list)
        # conn_reduced = []
        # for (a,b) in conn:
        #     essential = False
        #     if a in simplices_dict:
        #         if b in simplices_dict[a]:
        #             essential = True
        #     if essential:
        #         conn_reduced.append((a,b))
    
        
        return set(pts_reduced), conn # conn_reduced

    except:
        print("Reduction not possible")
        return pts_idx, conn      
  
def reduce3(pts_list, pts_idx, hp_pts):
    #try:
    pts_idx_list = list(pts_idx)
    pts = [pts_list[k] for k in pts_idx]
    hull = ConvexHull(pts, qhull_options = "")
    pts_reduced =  [l for (k,l) in enumerate(pts_idx_list) if k in hull.vertices]

    # Erstellen einer edges_reduced_list[k] alle Kanten des k-ten Simplex ent-
    # haelt
    conn_reduced_list = []
    for k,s in enumerate(hull.simplices):  
        s_idx_list = [pts_idx_list[k] for k in s]
        for a_idx, a in enumerate(s_idx_list):
            if a in hp_pts:
                for b in s_idx_list[a_idx+1:]:
                    if b in hp_pts:
                        conn_reduced_list.append(order((a,b)))


    hp_conn_reduced = set(conn_reduced_list)
    return set(pts_reduced), hp_conn_reduced

    #except:
     #   print("Reduction not possible")
      
      #  return pts_idx, conn

def reduce2(pts_list, pts_idx):
    #try:
    pts_idx_list = list(pts_idx)
    pts = [pts_list[k] for k in pts_idx]
    hull = ConvexHull(pts, qhull_options = "")
    pts_reduced =  [l for (k,l) in enumerate(pts_idx_list) if k in hull.vertices]

    # Erstellen einer edges_reduced_list[k] alle Kanten des k-ten Simplex ent-
    # haelt
    conn_reduced_list = []
    for k,s in enumerate(hull.simplices):  
        s_idx_list = [pts_idx_list[k] for k in s]
        conn_reduced_simplex_list = []
        for idx_a, a in enumerate(pts_idx_list):
            for b in pts_idx_list[idx_a+1:]:
                conn_reduced_simplex_list.append(a,b)
        conn_reduced_list.extend(conn_reduced_simplex_list)



        conn_reduced = set(conn_reduced_list)
        return set(pts_reduced), conn_reduced
    
    #except:
    #    print("Reduction not possible")
      
    #    return pts_idx, conn

def order(P):
    if P[0] <= P[1]:
        return P
    else:
        return (P[1], P[0])

    # if False:
    #         polytope_vertices = [np.append(np.ones(1), np.array(pt)) for pt in pts]
    #         mat = cdd.Matrix(polytope_vertices, number_type='float')
    #         mat.rep_type = cdd.RepType.GENERATOR
            
    #         poly_flag = True
    #         while poly_flag == True:
    #             try:
                    
                    
    #                 poly = cdd.Polyhedron(mat)
    #                 poly_flag = False
    #             except:
    #                 # print("except: regularize cdd matrix type 1")
    #                 regs[0] += 1
    #                 np_mat = np.array(mat)
    #                 np_mat_1 = np_mat[:,0]  
    #                 np_mat_2 = np_mat[:,1:]
    #                 # Regularisierung
    #                 np_mat_2 += (10**(-8))*np.ones(np_mat_2.shape)
    #                 np_mat_2 = np.column_stack((np_mat_1, np_mat_2))
                    
    #                 mat = cdd.Matrix(np_mat_2, number_type = 'float')
    #                 mat.rep_type = cdd.RepType.GENERATOR
    #         gens = poly.get_generators()
    #         np_gens = np.array(gens)
    #         if np_gens.shape[0] != 0:
    #             gens = np_gens[:,1:]
    #         print(len(gens), gens)
    #         print(len(polytope_vertices), gens)