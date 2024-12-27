from dash import dcc, html


def create_top_component(data_provider, data_provider_list):
    calculated_instances_selection_options = [
        {
        'label': ' | '.join(f'{x:.2f}' for x in value.base_instance),
        'value': index
        } for index, value in enumerate(data_provider_list)
    ]
    test_instances_selection_options = [{'label': i, 'value': i} for i in range(len(data_provider.test_instances))]
    class_names = data_provider.get_class_names()
    probas = data_provider.get_prediction()[0]
    feature_names = data_provider.feature_names
    component = html.Div(style={'width': '100%', 'display': 'flex', 'height': '100%'}, children=[
                    html.Div(style={'width': '70%', 'display': 'flex', 'flexDirection': 'column', 'flex': 1}, children=[
                        html.Div(style={'display': 'flex', 'justify-content': 'space-around', 'flex': 0.4}, children=[
                            html.Div(children=html.Div([
                                html.Label('datasets:'),
                                dcc.Dropdown(
                                    id='dataset-input',
                                    options=[{'label': 'iris', 'value': 'iris'},
                                             {'label': 'fetal health', 'value': 'fetal_health'}],
                                    value=data_provider.data_set_name,
                                    style={'width': '100%'}
                                ),
                            ]),
                                style={'flex': '0.5', 'padding': '5px', 'border': '1px solid #ccc',
                                       'box-sizing': 'border-box', 'text-align': 'center'}),
                            html.Div(children=html.Div([
                                html.Label('visualization width:'),
                                dcc.Input(
                                    id='width-input',
                                    type='number',
                                    value=data_provider.width,
                                    style={'width': '95%', 'height': '30px'}
                                ),
                            ]),
                                style={'flex': '0.5', 'padding': '5px', 'border': '1px solid #ccc',
                                       'box-sizing': 'border-box', 'text-align': 'center'}),
                            html.Div(children=[html.Div([

                                html.Label('test-instance-id'),
                                dcc.Dropdown(
                                    id='base-instance-id-input',
                                    options=test_instances_selection_options,
                                    value=data_provider.base_instance_id,
                                    style={'width': '100%', 'height': '100%'}  # Make dropdown fill its container
                                ),
                            ])],
                                     style={'flex': '1', 'padding': '5px', 'border': '1px solid #ccc',
                                            'box-sizing': 'border-box', 'text-align': 'center'}),


                            html.Div([
                                html.Label('saved instances'),
                                dcc.Dropdown(
                                    id='saved-data-provider-selection',
                                    options=calculated_instances_selection_options,
                                    style={'width': '100%', 'height': '100%'}
                                )
                            ], style={'flex': '1', 'padding': '5px', 'border': '1px solid #ccc',
                                            'box-sizing': 'border-box', 'text-align': 'center'}),


                            html.Div(children=html.Div([
                                html.Label('start calculation'),
                                html.Button('Start', id='start-calculation-button', n_clicks=0, style={'width': '100%', 'height': '35px'}),
                            ]),
                                     style={'flex': '1', 'padding': '5px', 'border': '1px solid #ccc',
                                            'box-sizing': 'border-box', 'text-align': 'center'}),
                            * [html.Div(children=[html.Div([

                                html.Label(f'prediction for: {class_names[i]}'),
                                html.Label(f'{val:.4f}', style={'height': '35px', 'paddingTop': '15px'}),

                            ], style={'flex': '1', 'padding': '5px', 'border': '1px solid #ccc',
                                      'box-sizing': 'border-box', 'text-align': 'center',
                                      'display': 'flex', 'flex-direction': 'column'})]) for i, val in enumerate(probas)]
                            #*[html.Div(children=f"Element {i+1}", style={'flex': '1', 'padding': '10px', 'border': '1px solid #ccc', 'box-sizing': 'border-box', 'text-align': 'center'}) for i in range(4)]
                        ]),
                        html.Div(style={'display': 'flex', 'justify-content': 'space-around', 'flex': 0.4}, children=[

                            html.Div([
                                html.Label('apply changes'),

                                html.Button(id='apply-changes-button', n_clicks=0, style={'height': '100%'}),
                            ], style={'display': 'flex', 'flex': '0.5', 'padding': '10px', 'border': '1px solid #ccc',
                                      'box-sizing': 'border-box', 'text-align': 'left', 'flexDirection': 'column'}),
                            *[
                                html.Div([
                                    html.Label(name),
                                    dcc.Input(
                                        id=f'feat-{index+1}',
                                        type='number',
                                        value=data_provider.base_instance[index]
                                    )
                                ], style={'display': 'flex', 'flex': '1', 'padding': '10px', 'border': '1px solid #ccc',
                                          'box-sizing': 'border-box', 'text-align': 'left', 'flexDirection': 'column'}) for index, name in enumerate(feature_names)
                            ],
                                     # *[html.Div(children=f"Element {i+1}", style={'flex': '1', 'padding': '10px', 'border': '1px solid #ccc', 'box-sizing': 'border-box', 'text-align': 'center'}) for i in range(4)]
                        ]),
                    ]),
                ])
    return component

