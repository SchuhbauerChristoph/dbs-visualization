from dash import dcc, html, dash_table
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.colors
import numpy as np

'''
this file contains all methods needed to create the information component and its content
'''


def create_information_component(data_provider, data_provider_list):
    tabs_creators = [
        lambda: create_chart_settings_tab(data_provider, data_provider_list),
        lambda: create_metrics_tab(data_provider.get_metrics_tab_data()),
        lambda: create_decision_boundary_tab(data_provider.get_additional_boundary_info()),
        lambda: create_feature_information_tab(data_provider.get_class_names())
    ]
    return html.Div(
        [
            html.Div(id='requested-dims-output', style={'marginTop': '0px'}),
            html.Div(id='ta', style={'marginTop': '0px'}, children=[
                create_tabs(tabs_creators)
            ]),
        ],
        style={'width': '100%', 'height': '100%'}
    )


def create_tabs(tab_creators):
    tabs = []
    for creator in tab_creators:
        try:
            tab = creator()
            if isinstance(tab, dcc.Tab):
                tabs.append(tab)
            else:
                print(f"Warning: Function {creator.__name__} did not return a dcc.Tab object.")
        except Exception as e:
            print(f"Warning: Error creating tab {creator.__name__}. Function:  Error: {e}")

    if tabs:
        return dcc.Tabs(children=tabs)
    else:
        print("Warning: No tabs could be created.")
        return None


def create_chart_settings_tab(data_provider, data_provider_list):
    data_provider_selection_dropdown_options = [
        {
            'label': ' | '.join(f'{x:.4f}' for x in value.base_instance),
            'value': index
        } for index, value in enumerate(data_provider_list)
    ]

    dimension_combinations = list(data_provider.get_dimension_combinations())
    dimensions_selection_dropdown_options = [{'label': key, 'value': f'{value}'} for key, value in
                                             dimension_combinations]

    class_names = data_provider.get_class_names()
    feature_names = data_provider.feature_names

    component = html.Div(style={'width': '98%', 'display': 'flex', 'flexDirection': 'column', 'flex': 1},
                         children=[
                             html.Div(style={'width': '100%', 'display': 'flex', 'flexDirection': 'column', 'flex': 1,
                                             'margin': '5px'},
                                      children=[
                                          html.Label('dimension-selection'),
                                          dcc.Checklist(
                                              id='charts-dimensions-selection',
                                              options=dimensions_selection_dropdown_options,
                                              value=[f'{val[1]}' for val in dimension_combinations[:4]],
                                              style={
                                                  "border": "1px solid #ccc",
                                                  "padding": "10px",
                                                  "borderRadius": "5px",
                                                  "maxHeight": "15vh",
                                                  "height": "15vh",
                                                  "overflowY": "auto"
                                              },
                                              inputStyle={"marginRight": "10px"},
                                              labelStyle={"display": "block", "paddingBottom": "5px",
                                                          "borderBottom": "1px solid #eee"}
                                          )

                                      ]),

                             html.Div(style={'width': '100%', 'display': 'flex', 'flexDirection': 'column', 'flex': 1,
                                             'margin': '5px'},
                                      children=[
                                          html.Label('instance-selection'),
                                          dcc.Checklist(
                                              id='charts-data-provider-selection',
                                              options=data_provider_selection_dropdown_options,
                                              style={
                                                  "border": "1px solid #ccc",
                                                  "padding": "10px",
                                                  "borderRadius": "5px",
                                                  "maxHeight": "15vh",
                                                  "height": "15vh",
                                                  "overflowY": "auto"
                                              },
                                              inputStyle={"marginRight": "10px"},
                                              labelStyle={"display": "block", "paddingBottom": "5px",
                                                          "borderBottom": "1px solid #eee"}
                                          )

                                      ]),

                             html.Div(style={'width': '100%', 'display': 'flex', 'flexDirection': 'column', 'flex': 1,
                                             'margin': '5px'},
                                      children=[
                                          html.Label('additional visualization options'),
                                          dcc.Dropdown(
                                              id='charts-optional-element-selection',
                                              options=[
                                                  {'label': 'class area', 'value': 'z1'},
                                                  *[{'label': f'probability for: {val}', 'value': f'z2_{index}'}
                                                    for index, val in enumerate(class_names)],
                                                  *[{'label': f'partial dependancy plot - {val}',
                                                     'value': f'z3_{index}'} for index, val in
                                                    enumerate(class_names)],
                                                  *[{'label': f'feature values for: {val}', 'value': f'z5_{index}'} for
                                                    index, val in enumerate(feature_names)],
                                              ],
                                              value='z1',
                                          ),
                                      ]),
                             html.Div(style={'width': '100%', 'display': 'flex', 'flexDirection': 'column', 'flex': 1,
                                             'margin': '5px'},
                                      children=[
                                          html.Button('update plots', id='update-charts-button', n_clicks=0,
                                                      style={'height': '35px', 'margin': '0px'}),
                                      ]),
                         ])
    return dcc.Tab(label='plot settings', children=component, style={'backgroundColor': '#f0f0f0'})


def create_metrics_tab(metrics_data):
    metrics_data_df, complete_mat, sub_mat = metrics_data
    info_elements = []

    info_elements.append(html.Div(
        [dcc.Graph(figure=complete_mat)],
        style={'border': '1px solid #ccc', 'padding': '3px', 'margin': '3px'}
    ))
    info_elements.append(html.Div(
        [dcc.Graph(figure=sub_mat)],
        style={'border': '1px solid #ccc', 'padding': '3px', 'margin': '3px'}
    ))

    scrollable_content = html.Div(
        info_elements,
        style={
            'maxHeight': '70vh',
            'overflowY': 'auto',
            'padding': ' 0px'
        }
    )

    return dcc.Tab(label='model evaluation metrics', children=scrollable_content, style={'backgroundColor': '#f0f0f0'})


def create_decision_boundary_tab(table_data):
    if not table_data:

        datatable = html.P("No data to display in the table.")
        div = html.Div([datatable], style={'border': '1px solid #ccc', 'padding': '5px', 'margin': '5px'})
    else:
        columns = [{'name': k, 'id': k} for k in table_data[0].keys()]
        datatable = dash_table.DataTable(
            table_data,
            columns,
            style_data={'width': '100%'},
            style_cell={'textAlign': 'left'},
            style_header={'backgroundColor': 'paleturquoise'},
            page_size=10,
        )
        div = html.Div([datatable], style={'border': '1px solid #ccc', 'padding': '5px', 'margin': '5px'})

    return dcc.Tab(label='decision boundary information', children=[div], style={'backgroundColor': '#e0e0e0'})


def create_feature_information_tab(class_names):
    dropdown_options = [
        {'label': 'feature distribution: training-data', 'value': '1'},
        {'label': 'correlation matrix', 'value': '2'},
        *[{'label': f'shap summary chart ({name})', 'value': f'{index + 3}'} for index, name in enumerate(class_names)],
    ]
    drop_div = html.Div(
        [
                            dcc.Dropdown(
                                id='feature_info_dropdown',
                                options=dropdown_options,
                                value='1',
                                style={'width': '100%'}
                            )],
        style={'border': '1px solid #999', 'padding': '3px', 'margin': '3px', 'display': 'inline-block',
               'width': '97%'}
    )

    content_div = html.Div(id='feature-info-content-output',
            style={'border': '1px solid #999', 'padding': '3px', 'margin': '3px', 'display': 'inline-block', 'width':'97%'
               }
    )

    return dcc.Tab(label='feature information', children=[drop_div, content_div],
                   style={'backgroundColor': '#e0e0e0'})


def set_feature_information_tab_content(dropdown_value, data_provider):
    if dropdown_value == '1':
        figs = [create_kde(data_provider, x) for x in range(len(data_provider.base_instance))]

        charts = [dcc.Graph(figure=fig) for fig in figs]
        scrollable_content = html.Div(
            charts,
            style={
                'maxHeight': '64vh',
                'overflowY': 'auto',
                'padding': '0px'
            }
        )
        return html.Div(children=[html.P(''), scrollable_content])

    if dropdown_value == '2':
        corr_matrix_fig = data_provider.get_correlation_matrix()
        return html.Div(children=[dcc.Graph(figure=corr_matrix_fig)])

    if int(dropdown_value) > 2:
        class_idx = int(dropdown_value) - 3
        shap_values = data_provider.shap_values
        shap_fig = create_shap_summary_plot_for_single_class(shap_values, data_provider.feature_names, class_idx, data_provider.train_data, data_provider.base_instance_id)
        return html.Div(children=[html.P(''), dcc.Graph(figure=shap_fig)])


def create_kde(data_provider, index):

    feature_names = data_provider.feature_names
    train_data = data_provider.train_data
    data = train_data
    train_label = data_provider.y_train_data
    labels = train_label

    index = index

    class_names = data_provider.get_class_names()

    feature_data = data[:, index]

    kde_hist_data = []
    kde_group_labels = []

    single_value_points = []
    single_value_labels = []

    unique_values = np.unique(feature_data)
    if len(unique_values) > 1:
        kde_hist_data.append(feature_data)
        kde_group_labels.append("All")
    elif len(unique_values) == 1:
        single_value_points.append(unique_values[0])
        single_value_labels.append("All")

    for class_idx, class_name in enumerate(class_names):
        class_feature_data = feature_data[labels == class_idx]
        if class_feature_data.size == 0:
            continue
        unique_values = np.unique(class_feature_data)
        if len(unique_values) > 1:
            kde_hist_data.append(class_feature_data)
            kde_group_labels.append(class_name)
        elif len(unique_values) == 1:
            single_value_points.append(unique_values[0])
            single_value_labels.append(class_name)

    colors = plotly.colors.DEFAULT_PLOTLY_COLORS

    if len(kde_hist_data) > 0:
        fig = ff.create_distplot(kde_hist_data, kde_group_labels, show_hist=False, show_rug=False,
                                 colors=colors[:len(kde_group_labels)])
    else:
        fig = go.Figure()

    for idx, (x_val, label) in enumerate(zip(single_value_points, single_value_labels)):
        color_idx = kde_group_labels.index(label) if label in kde_group_labels else len(kde_group_labels) + idx
        color = colors[color_idx % len(colors)]

        fig.add_vline(
            x=x_val,
            line=dict(color=color, dash='dot'),
            name=f"{label} (single value)",
            annotation_text=None
        )
        fig.add_trace(
            go.Scatter(
                x=[x_val, x_val],
                y=[0, 1],
                mode="lines",
                line=dict(color=color, dash="dot"),
                name=f"{label} (single value)",
                showlegend=True,
            )
        )

    predicted_instance = data_provider.base_instance[index]
    fig.add_trace(go.Scatter(
        x=[predicted_instance],
        y=[None],
        mode='lines',
        marker=dict(size=10, color="black"),
        name="Predicted Instance",
        showlegend=True
    ))

    fig.update_layout(
        shapes=[
            {
                'type': 'line',
                'x0': predicted_instance,
                'y0': 0,
                'x1': predicted_instance,
                'y1': 1,
                'xref': 'x',
                'yref': 'paper',
                'line': {'color': 'black', 'width': 2}
            }
        ]
    )

    fig.update_layout(
        title=f'KDE Plot for {feature_names[index]}',
        xaxis_title='Feature Value',
        yaxis_title='Density',
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    return fig


def create_shap_summary_plot_for_single_class(shap_values, features, class_index, instances, highlight_instance=None):
    num_instances, num_features, num_classes = shap_values.shape

    fig = go.Figure()

    for feature_idx in range(num_features):
        feature_name = features[feature_idx]
        feature_shap_vals = shap_values[:, feature_idx, class_index]

        y_range = (-20, 20)
        if y_range is not None:
            ymin, ymax = y_range
            feature_shap_vals_normalized = (feature_shap_vals - np.min(feature_shap_vals)) / (np.max(feature_shap_vals) - np.min(feature_shap_vals))
            feature_shap_vals = ymin + feature_shap_vals_normalized * (ymax - ymin)

        feature_name = feature_name[:15] + "..." if len(feature_name) > 15 else feature_name
        fig.add_trace(go.Scatter(
            x=feature_shap_vals,
            y=[feature_name] * len(feature_shap_vals),
            mode='markers',
            marker=dict(
                size=6,
                color='blue',
                colorscale='Plasma',
                showscale=False,
            ),
            name=f'Class {class_index + 1} - {feature_name}',
            orientation='v',
            showlegend=False
        ))

        if highlight_instance is not None:
            fig.add_trace(go.Scatter(
                x=[feature_shap_vals[highlight_instance]],
                y=[feature_name],
                mode='markers',
                marker=dict(
                    size=12,
                    color='gold',
                    symbol='star',
                    line=dict(width=2, color='black')
                ),
                name=f'Instance {highlight_instance + 1}',
                orientation='v',
                showlegend=False
            ))

    fig.update_layout(
        xaxis_title="SHAP Value",
        yaxis_title="Feature",
        height=550,
    )

    fig.update_layout(yaxis=dict(tickangle=-45))
    return fig
