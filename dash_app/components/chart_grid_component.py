import numpy as np
import plotly.graph_objects as go
from dash import dcc, html
from dataProvider import DataProvider

'''
this file contains all methods needed for creating the chart component or the charts themself
'''
def create_chart_grid_component(data_provider_list=None):

    all_figure_data = []
    dims_idx = 0

    for provider in data_provider_list:
        dat = provider.get_all_figure_data()
        all_figure_data.append(dat[dims_idx])

    figures, colors = createAllCharts(all_figure_data)

    return html.Div(
        [
            html.Div(
                id='chart-grid-container',
                children=[
                    dcc.Graph(id=f'chart-1', figure=figures[0], style={'margin': '2px'}),
                    dcc.Graph(id=f'chart-2', figure=figures[1], style={'margin': '2px'}),
                    dcc.Graph(id=f'chart-3', figure=figures[2], style={'margin': '2px'}),
                    dcc.Graph(id=f'chart-4', figure=figures[3], style={'margin': '2px'}),
                    dcc.Graph(id=f'chart-5', figure=figures[4], style={'margin': '2px'}),
                    dcc.Graph(id=f'chart-6', figure=figures[5], style={'margin': '2px'}),
                ],
                style={
                    'display': 'grid',
                    'grid-template-columns': 'repeat(3, 1fr)',
                    'grid-template-rows': 'repeat(2, 1fr)',
                    'grid-gap': '5px',
                    'height': '95%'
                }
            ),
        ],
        style={'width': '100%', 'height': '100%'}
    )


def createAllCharts(all_figure_data, highlight_instance=None, heatmap_select='z1'):
    figures = []
    og_colors = []
    for data in all_figure_data:

        temp, cols = createFig(data, highlight_instance, heatmap_select)
        figures.append(temp)
        og_colors.append(cols)

    while len(figures) < 6:
        figures.extend([go.Figure()])

    charts = []
    for i, fig in enumerate(figures):
        charts.append(dcc.Graph(id=f'chart-{i}', figure=fig))

    colors = []
    for i, color in enumerate(og_colors):
        colors.append(html.Div(id=f'original-colors-{i}', style={'display': 'none'}, children=color))

    return figures, colors


def createFig(figure_data: dict, highlight_instance=None, heatmap_select='z1'):
    instance = figure_data['instance']
    line_data = figure_data['line_data']
    relevant_features = figure_data['relevant_features']
    instances_to_add = figure_data['instances_to_add']
    raw_prediction_near_instances = figure_data['prediction_near_instances']
    prediction_near_instances = np.argmax(raw_prediction_near_instances, axis=1)
    points = figure_data['points']
    all_probs = figure_data['all_probs']
    class_colors = figure_data['class_colors']

    feat_name_x, feat_name_y = figure_data['feature_names']

    x_feature, y_feature = relevant_features
    x_values = [point[x_feature] for point in points]
    y_values = [point[y_feature] for point in points]

    limits = [x_values[0], x_values[-1], y_values[0], y_values[-1]]

    heat_map: go.Heatmap = create_class_area_heatmap(points, all_probs, relevant_features, class_colors, len(class_colors))

    if heatmap_select.startswith('z2_'):
        index = heatmap_select.split('_')[1]
        index = int(index)
        heat_map = create_probability_heatmap_by_class(points, all_probs, relevant_features, index)

    if heatmap_select.startswith('z3_'):
        feat_name_1 = f'feature{relevant_features[0] + 1}'
        feat_name_2 = f'feature{relevant_features[1] + 1}'

        feat_name_1 = feat_name_x
        feat_name_2 = feat_name_y

        pdp_info_dict = figure_data['pdp_data']
        try:
            feat1_2_data = pdp_info_dict[tuple([feat_name_1, feat_name_2])]
            index = heatmap_select.split('_')[1]
            index = int(index)
            heat_map = create_pdp_heatmap_by_class(feat1_2_data, index, limits)
        except Exception as e:
            raise ValueError(f'pdp could not be calculated for feature pair: {feat_name_1} / {feat_name_2}. reason: insufficient variability')


    labels = instances_to_add[1]
    instance_label = figure_data['instance_label']

    colors = [class_colors[int(label)] for label in labels]
    nears_prediction_colors = [class_colors[pred] for pred in prediction_near_instances]

    if heatmap_select.startswith('z5_'):
        index = heatmap_select.split('_')[1]
        index = int(index)

        first_values = [instance[index] for instance in instances_to_add[0]]
        normalized_values = (np.array(first_values) - min(first_values)) / (
                max(first_values) - min(first_values)) if max(first_values) - min(
            first_values) != 0 else np.zeros_like(first_values)
        colors = [value for value in normalized_values]
        nears_prediction_colors = colors

    instance_color = [class_colors[int(instance_label)]]
    if highlight_instance is not None:
        index = None
        for i, instance_ in enumerate(instances_to_add[0]):
            res = instance_ == highlight_instance
            if all(res):
                index = i

        if index is not None:
            colors[index] = 'white'

    near_instances_trace = create_scatter_trace(instances_to_add, relevant_features, colors, nears_prediction_colors)
    if heatmap_select.startswith('z5_'):
        heat_map.showscale = False
        near_instances_trace.marker.colorbar = dict(title='')

    base_instance_trace = create_scatter_trace([[instance], None], relevant_features, instance_color, ['green'], _symbol='x')

    fig = go.Figure(data=[heat_map])
    fig.add_trace(near_instances_trace)
    fig.add_trace(base_instance_trace)

    decision_boundary_traces = create_decision_boundary_traces(line_data)
    for trace in decision_boundary_traces:
        fig.add_trace(trace)

    fig.update_layout(
        title={
            'text': ' | '.join(f'{x:.4f}' for x in instance),
            'font': dict(
                size=12
            ),
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis=dict(range=[min(x_values) - 0.01, max(x_values) + 0.01], constrain='domain'),
        yaxis=dict(range=[min(y_values) - 0.01, max(y_values) + 0.01], scaleanchor="x", scaleratio=1),
        xaxis_title=feat_name_x,
        yaxis_title=feat_name_y,
        margin=dict(l=60, r=60, t=60, b=60)
    )
    return fig, colors


def create_pdp_heatmap_by_class(pdp_interact, class_idx, limits):

    pdp_results = pdp_interact.results[class_idx]
    feat1_name, feat2_name = pdp_interact.feature_names
    feat1_values = [combo[0] for combo in pdp_interact.feature_grid_combos]
    feat2_values = [combo[1] for combo in pdp_interact.feature_grid_combos]
    values = pdp_results.pdp

    feat1_values_filtered = []
    feat2_values_filtered = []
    values_filtered = []

    feat1_min = limits[0]
    feat1_max = limits[1]
    feat2_min = limits[2]
    feat2_max = limits[3]

    for x, y, z in zip(feat1_values, feat2_values, values):
        if feat1_min <= x <= feat1_max and feat2_min <= y <= feat2_max:
            feat1_values_filtered.append(x)
            feat2_values_filtered.append(y)
            values_filtered.append(z)

    map = go.Heatmap(
        z=values_filtered,
        x=feat1_values_filtered,
        y=feat2_values_filtered,
        colorscale='Viridis',
        opacity=0.4,
        zmin=0,
        zmax=1

    )
    return map


def create_probability_heatmap_by_class(grid_points, values, relevant_features, class_idx):
    x_feature, y_feature = relevant_features
    x_values = [point[x_feature] for point in grid_points]
    y_values = [point[y_feature] for point in grid_points]

    xi = np.unique(x_values)
    yi = np.unique(y_values)
    grid_z = np.zeros((len(yi), len(xi)))

    for point, value in zip(grid_points, values):
        x_index = np.where(xi == point[x_feature])[0][0]
        y_index = np.where(yi == point[y_feature])[0][0]
        grid_z[y_index, x_index] = value[class_idx]

    return go.Heatmap(
        z=grid_z,
        x=xi,
        y=yi,
        zmin=0,
        zmax=1,
        colorscale='Viridis',
        opacity=0.4
    )


def create_class_area_heatmap(grid_points, all_probs, relevant_features, colors, n_classes: int):

    x_feature, y_feature = relevant_features
    x_values = [point[x_feature] for point in grid_points]
    y_values = [point[y_feature] for point in grid_points]

    xi = np.unique(x_values)
    yi = np.unique(y_values)
    grid_z = np.zeros((len(yi), len(xi)))

    temp_classes = np.argmax(all_probs, axis=1)
    classes = temp_classes.reshape(-1, 1).tolist()

    for point, value in zip(grid_points, classes):
        x_index = np.where(xi == point[x_feature])[0][0]
        y_index = np.where(yi == point[y_feature])[0][0]
        grid_z[y_index, x_index] = value[0]

    return go.Heatmap(
        z=grid_z,
        x=xi,
        y=yi,
        zmax=n_classes - 1,
        zmin=0,
        colorscale=colors,
        showscale=True,
        opacity=0.4
    )


def create_decision_boundary_traces(line_data):

    traces = []

    for line_segment in line_data:
        x_values = [line_segment[0][0], line_segment[1][0]]
        y_values = [line_segment[0][1], line_segment[1][1]]

        trace = go.Scatter(
            x=x_values,
            y=y_values,
            mode='lines',
            line=dict(color='red'),
            showlegend=False
        )
        traces.append(trace)
    return traces


def create_scatter_trace(instances, relevant_features, label_color, prediction_colors, _symbol='circle'):
    instances = instances[0]

    x_feature, y_feature = relevant_features
    x_values = [instance[x_feature] for instance in instances]
    y_values = [instance[y_feature] for instance in instances]

    customdata = [{
        'trace_type': 'scatter_1',
        'instance_data': instances
    }]

    if _symbol == 'x':
        return go.Scatter(
                x=x_values,
                y=y_values,
                mode='markers',
                marker=dict(
                    color=label_color,
                    symbol=_symbol,
                    size=15,
                    line=dict(
                        color='white',
                        width=1
                    )
                ),
                showlegend=False,
                customdata=customdata,
            )

    return go.Scatter(
            x=x_values,
            y=y_values,
            mode='markers',
            marker=dict(
                color=label_color,
                symbol=_symbol,
                size=10,
                line=dict(
                    color=prediction_colors,
                    width=2
                )
            ),
            showlegend=False,
            customdata=customdata,
        )


def create_shap_visuals(limits, dataframe, feat1_idx, feat2_idx, class_idx):

    feat1_min = limits[0]
    feat1_max = limits[1]
    feat2_min = limits[2]
    feat2_max = limits[3]

    feat1_column_name = dataframe.columns[feat1_idx]
    feat2_column_name = dataframe.columns[feat2_idx]

    filtered_df = dataframe[(dataframe[feat1_column_name] >= feat1_min) & (dataframe[feat1_column_name] <= feat1_max) &
                     (dataframe[feat2_column_name] >= feat2_min) & (dataframe[feat2_column_name] <= feat2_max) & (dataframe['class_idx'] == class_idx)]

    trace = go.Scatter(
        x=filtered_df[feat1_column_name],
        y=filtered_df[feat2_column_name],
        mode='markers',
        marker=dict(
            size=20,
            color=filtered_df['value'],
            colorscale='Inferno',
            showscale=True
        ),
        showlegend=False,
    )
    return trace






