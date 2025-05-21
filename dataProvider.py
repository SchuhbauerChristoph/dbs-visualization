import itertools
from copy import deepcopy

import pdpbox.pdp

from decision_boundaries_v2 import calculate_decision_boundaries, createGrid, load_dataset_from_updated_path, get_shap_values, \
    load_model_from_updated_path, model_wrapper
import numpy as np
from typing import List
import pandas as pd
from pdpbox import pdp
import plotly.graph_objects as go
import matplotlib.colors as mcolors
alpha = 0

'''
an entity of the data provider class contains all information related to a single instance. 
a new data provider is created for each instance during the first time it is calculated
the job of the data provider is to hold the information and provide it to the other components
'''
class DataProvider():
    def __init__(self):

        #self.data_set_name = 'fetal_health'
        self.data_set_name = 'iris'
        train_data, y_train_data, test_data, y_test_data, test_instances, test_label = load_dataset_from_updated_path(self.data_set_name)
        self.model = load_model_from_updated_path(self.data_set_name)
        self.base_instance_id: int = 4
        self.step_size: float = 0.01
        self.width: float = 0.5
        self.train_data = train_data
        self.y_train_data = y_train_data
        self.test_data = test_data
        self.y_test_data = y_test_data
        self.test_instances = test_instances
        self.test_label = test_label
        self.feature_names = self.set_feature_names()
        self.base_instance: List[float] = test_instances[self.base_instance_id]
        self.temp_feature_values = [None for x in range(len(self.base_instance))]
        for index, x in enumerate(self.temp_feature_values):
            if x is not None:
                self.base_instance[index] = x
        self.near_instances = None
        self.boundaries = None
        self.lower_ends = None
        self.upper_ends = None
        self.fact_dict = None
        self.grid_points_dict = None
        self.grid_probs_dict = None
        self.grid_all_probs_dict = None
        self.requested_dims = list(itertools.combinations(list(range(len(self.base_instance))), 2))
        self.shap_values = None
        self.set_boundary_data()
        self.set_grid_data()
        self.set_shap_values()
        self.pdp_info_dict = self.calc_pdp_values()
        self.shap_df = self.get_shap_info()
        self.class_colors = self.set_class_colors()

    def change_dataset(self, dataset_name):
        self.data_set_name = dataset_name
        self.class_colors = self.set_class_colors()
        train_data, y_train_data, test_data, y_test_data, test_instances, test_label = load_dataset_from_updated_path(self.data_set_name)
        self.model = load_model_from_updated_path(self.data_set_name)
        self.train_data = train_data
        self.y_train_data = y_train_data
        self.test_data = test_data
        self.y_test_data = y_test_data
        self.test_instances = test_instances
        self.test_label = test_label
        self.feature_names = self.set_feature_names()
        self.base_instance: List[float] = test_instances[self.base_instance_id]
        for index, x in enumerate(self.temp_feature_values):
            if x is not None:
                self.base_instance[index] = x
        self.near_instances = None
        self.boundaries = None
        self.lower_ends = None
        self.upper_ends = None
        self.fact_dict = None
        self.grid_points_dict = None
        self.grid_probs_dict = None
        self.grid_all_probs_dict = None
        self.requested_dims: List[int] = list(itertools.combinations(list(range(len(self.base_instance))), 2))
        self.shap_values = None
        self.set_boundary_data()
        self.set_grid_data()
        self.set_shap_values()
        self.pdp_info_dict = self.calc_pdp_values()
        self.shap_df = self.get_shap_info()
        self.class_colors = self.set_class_colors()

    def set_boundary_data(self):
        [vis_instance, vis_inst_data, near_train_insts, boundaries, extra_data] = calculate_decision_boundaries(
            self.base_instance_id,
            self.width,
            self.test_instances,
            self.test_label,
            self.train_data,
            self.y_train_data,
            self.model,
            self.base_instance
        )

        self.upper_ends = vis_inst_data[0]
        self.lower_ends = vis_inst_data[1]
        self.near_instances = near_train_insts
        self.boundaries = boundaries
        self.fact_dict = extra_data

    def set_grid_data(self):
        varibleFeatures = list(itertools.combinations(list(range(len(self.base_instance))), 2))
        grid_points_dict = {}
        grid_probs_dict = {}
        grid_all_probs_dict = {}

        for dims in varibleFeatures:
            temp_points, temp_probas, temp_all_probs = createGrid(self.base_instance, dims, self.lower_ends, self.upper_ends, self.step_size, self.model)

            grid_points_dict[tuple(dims)] = temp_points
            grid_probs_dict[tuple(dims)] = temp_probas
            grid_all_probs_dict[tuple(dims)] = temp_all_probs

        self.grid_points_dict: dict = grid_points_dict
        self.grid_probs_dict: dict = grid_probs_dict
        self.grid_all_probs_dict: dict = grid_all_probs_dict

    def set_shap_values(self):
        shap_values, expected_values = get_shap_values(self.train_data, self.y_train_data, self.test_data, self.y_test_data, self.model)
        self.shap_values = shap_values

    def redo_calculation(self, dataset_name):
        if dataset_name != self.data_set_name:
            self.change_dataset(dataset_name)
            return
        self.base_instance: List[float] = self.test_instances[self.base_instance_id]
        for index, x in enumerate(self.temp_feature_values):
            if x is not None:
                self.base_instance[index] = x

        self.set_boundary_data()
        self.set_grid_data()
        self.pdp_info_dict = self.calc_pdp_values()
        self.shap_df = self.get_shap_info()

    def set_requested_dims(self, selected_values):
        if len(selected_values) == 0:
            self.requested_dims = list(itertools.combinations(list(range(len(self.base_instance))), 2))
        else:
            self.requested_dims = sorted(selected_values)

    def get_all_figure_data(self):
        figure_data_list = []
        dimension_combs = self.requested_dims
        near_instance_probs = model_wrapper(self.near_instances[0], self.model)
        for comp in dimension_combs:
            figure_data = {
                'dims': str(comp),
                'instance': self.base_instance,
                'instance_label': self.test_label[self.base_instance_id],
                'line_data': self.boundaries[comp],
                'relevant_features': comp,
                'instances_to_add': self.near_instances,
                'prediction_near_instances': near_instance_probs,
                'points': self.grid_points_dict[comp],
                'probs': self.grid_probs_dict[comp],
                'all_probs': self.grid_all_probs_dict[comp],
                'feature_names': self.get_feature_names(comp),
                'lower_bounds': self.lower_ends,
                'upper_bounds': self.upper_ends,
                'pdp_data': self.pdp_info_dict,
                'shap_df': self.shap_df,
                'class_colors': self.class_colors
            }
            figure_data_list.append(figure_data)
        if len(figure_data_list) < 7:
            return figure_data_list
        else:
            return figure_data_list[:6]


    def get_class_names(self):
        if self.data_set_name == 'iris':
            return ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        return ['neutral', 'dangerous']

    def set_class_colors(self):
        low_color = 'purple'
        high_color = 'yellow'
        number_of_classes = 3 if self.data_set_name == 'iris' else 2

        cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', [low_color, high_color], N=number_of_classes)
        colors = [cmap(i/number_of_classes) for i in range(number_of_classes)]
        hex_colors = [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" for r, g, b, a in colors]
        return hex_colors

    def set_feature_names(self):
        if self.data_set_name == 'iris':
            feature_names = [
                'sepal length',
                'sepal width',
                'petal length',
                'petal width'
            ]
        else:
            feature_names = [
                'baseline value',
                'accelerations',
                'fetal_movement',
                'uterine_contractions',
                'prolongued_decelerations',
                'abnormal_short_term_variability',
                'histogram_width'
            ]
        return feature_names

    def get_correlation_matrix(self):
        corr_ = np.corrcoef(self.near_instances[0], rowvar=False)
        corr = corr_.copy()
        fig = go.Figure(data=go.Heatmap(
            z=corr,
            x=self.feature_names,
            y=self.feature_names,
            colorscale='Plasma',
            zmin=-1, zmax=1
        ))
        fig.update_layout(title='Correlation Matrix')
        fig.update_layout(
            title='Correlation Matrix',
            xaxis=dict(tickangle=30),
            yaxis=dict(tickangle=-50)
        )
        return fig

    def get_prediction(self):
        pred = self.model.propagate([self.base_instance])
        all_logits = np.array(pred)
        probabilities = np.exp(all_logits) / np.sum(np.exp(all_logits), axis=1, keepdims=True)
        return probabilities

    def get_additional_boundary_info(self):
        temp = []
        for k, v in self.fact_dict.items():
            v = v.round(6)

            def truncate_text(text, length=13):
                return text[:length] + "..." if len(text) > length else text

            dims_name = f'{truncate_text(self.feature_names[k[0]])}/{truncate_text(self.feature_names[k[1]])}'
            t = {'dimensions': f'{dims_name}', 'overall distance': v.iloc[3,0], 'nearest DB point': f'{v.iloc[1,0]} / {v.iloc[1,1]}'}
            temp.append(t)
        return temp

    def get_feature_names(self, dims):
        return self.feature_names[dims[0]], self.feature_names[dims[1]]

    def get_dimension_combinations(self):
        vals = list(itertools.combinations(list(range(len(self.base_instance))), 2))
        labels = [f'{self.feature_names[x]} | {self.feature_names[y]}' for [x, y] in vals]
        return zip(labels, vals)

    def get_metrics_tab_data(self):
        instances = self.test_instances
        labels = self.test_label
        _predictions = model_wrapper(instances, self.model)
        predictions = [np.argmax(x) for x in _predictions]

        from metrics import met2_b
        res = met2_b(
            data_train=self.train_data,
            y_train=self.y_train_data,
            instances=[self.base_instance],
            model=self.model,
            radius=0.1,
            mode="cube"
        )
        subset_conf_matrix = res[0]

        unique_classes = np.unique(labels)
        n_classes = len(unique_classes)
        complete_conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
        for true, pred in zip(labels, predictions):
            true = int(true)
            complete_conf_matrix[true, pred] += 1

        class_names = self.get_class_names()
        def create_conf_matrix_heatmap(matrix, title):
            fig = go.Figure(
                data=go.Heatmap(
                    z=matrix,
                    x=[f"Predicted {name}" for name in class_names],
                    y=[f"True {name}" for name in class_names],
                    colorscale='Viridis',
                    showscale=True,
                    zmin=0,
                )
            )
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    fig.add_annotation(
                        text=str(matrix[i, j]),
                        x=j,
                        y=i,
                        font=dict(color="white" if matrix[i, j] < 4 else 'black'),
                        showarrow=False
                    )
            fig.update_layout(
                title=title,
                template="plotly_white"
            )
            return fig

        metrics_df = None
        complete_conf_matrix_fig = create_conf_matrix_heatmap(complete_conf_matrix, "confusion matrix (all test instances)")
        subset_conf_matrix_fig = create_conf_matrix_heatmap(subset_conf_matrix, "confusion matrix (close test instances)")
        return metrics_df, complete_conf_matrix_fig, subset_conf_matrix_fig

    def calc_pdp_values(self):
        train = self.train_data
        column_names = self.feature_names
        df = pd.DataFrame(train, columns=column_names)

        from itertools import permutations
        feat_combs = list(permutations(column_names, 2))
        results_dict = {}

        for feat1, feat2 in feat_combs:
            feat1_idx = column_names.index(feat1)
            feat2_idx = column_names.index(feat2)

            feature1_min = self.lower_ends[feat1_idx]
            feature1_max = self.upper_ends[feat1_idx]
            feature2_min = self.lower_ends[feat2_idx]
            feature2_max = self.upper_ends[feat2_idx]

            X_train = df[
                (df[feat1] >= feature1_min) & (df[feat1] <= feature1_max) &
                (df[feat2] >= feature2_min) & (df[feat2] <= feature2_max)
                ]

            preds = self.model.predict_proba(X_train.to_numpy())
            firsts = []
            for x in preds:
                if x[0] > x[1]:
                    firsts.append(x)

            if len(X_train) < 2:
                print(f'skip {feat1} and {feat2} because length')
                continue

            if X_train[feat1].nunique() < 2 or X_train[feat2].nunique() < 2:
                print(f"insufficient variability for features ({feat1}, {feat2}) in filtered data. Skipping.")
                print(f'skip {feat1} and {feat2} because unique')
                continue

            features = [feat1, feat2]
            pdp_interact = pdp.PDPInteract(
                model=self.model,
                df=X_train,
                model_features=X_train.columns,
                features=features,
                feature_names=features,
                n_classes=len(self.get_class_names())
            )

            # ---------------------------------------------------------------------
            # add the missing “negative” curve for binary models
            # PDPbox docs: for binary tasks `pdp_interact.results` contains ONE PDPResult
            # (the positive‐class curve).
            # ---------------------------------------------------------------------
            if (
                    len(self.get_class_names()) == 2
                    and len(pdp_interact.results) == 1
            ):
                pos_res = pdp_interact.results[0]
                neg_res = deepcopy(pos_res)

                neg_res.pdp = 1.0 - pos_res.pdp
                neg_res.class_id = self.get_class_names()[0]
                pos_res.class_id = self.get_class_names()[1]

                pdp_interact.results = [neg_res, pos_res]
                pdp_interact.n_classes = 2
            results_dict[(feat1, feat2)] = pdp_interact

        return results_dict

    def get_shap_info(self):
        feature_columns = [f'feat{i + 1}' for i in range(len(self.test_instances[0]))]
        dataframe = pd.DataFrame(columns=feature_columns + ['feature_idx', 'class_idx', 'value'])

        for i, instance in enumerate(self.test_instances):
            feature_values = {feature_columns[k]: value for k, value in enumerate(instance)}
            shap_value = self.shap_values[i]

            for j, feature in enumerate(shap_value):
                for class_idx, value in enumerate(feature):
                    row = pd.Series({**feature_values, 'feature_idx': j, 'class_idx': class_idx, 'value': value})
                    dataframe.loc[len(dataframe)] = row
        return dataframe





