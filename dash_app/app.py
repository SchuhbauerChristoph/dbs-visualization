import dash
from dash import html
from dash.dependencies import Input, Output, State
from dash_app.components.chart_grid_component import create_chart_grid_component, createAllCharts
from dash_app.components.information_component import create_information_component, set_feature_information_tab_content
from dash import no_update, dcc
from dash_app.components.settings_component import create_settings_component
from dataProvider import DataProvider
import plotly.graph_objects as go
from typing import List
import copy

'''
this file contains the base configuration of the app and all the callbacks
'''

data_provider = DataProvider()
data_provider_list = [data_provider]
data_provider = data_provider_list[0]
app = dash.Dash(__name__)

app.layout = html.Div(
    children=[
        html.Div(id='hidden-div', style={'display': 'none'}),
        html.Div(
            style={
                'width': '100%',
                'height': '15%',
                'border': '1px solid black',
            },
            id='top-area',
            children=create_settings_component(data_provider, data_provider_list)
        ),

        html.Div(
            style={
                'width': '100%',
                'height': '85%',
                'border': '1px solid black',
                'display': 'flex'
            },
            children=[
                html.Div(
                            style={
                                'width': '70%',
                                'height': '100%',
                                'border': '1px solid black',
                                'display': 'flex',
                                'flex-direction': 'column'
                            },
                            id='chart-grid-area',
                            children=create_chart_grid_component(data_provider_list)
                ),
                html.Div(
                            style={
                                'width': '30%',
                                'height': '100%',
                                'border': '1px solid black',
                                'display': 'flex',
                                'flex-direction': 'column'
                            },
                            id='tabs-area',
                            children=create_information_component(data_provider, data_provider_list)
                ),
            ]
        ),
    ],
    style={'height': '100vh', 'display': 'flex', 'flex-direction': 'column'}
)


def update_chart_grid(n_click, selected_values, heatmap_select, alt_figure_data=None) -> List[go.Figure]:
    if n_click is None:
        return [no_update for i in range(6)]

    if selected_values:
        data_provider.set_requested_dims(selected_values)
    else:
        return [no_update for i in range(6)]

    figure_data = data_provider.get_all_figure_data()
    if alt_figure_data is not None and len(alt_figure_data) != 0:
        figure_data = alt_figure_data
    updated_charts, colors = createAllCharts(figure_data, heatmap_select=heatmap_select)

    return updated_charts


@app.callback(
    *[Output(f'chart-{i+1}', 'figure') for i in range(6)],
    Output('top-area', 'children'),
    Output('chart-grid-area', 'children'),
    Output('tabs-area', 'children'),
    Output('saved-data-provider-selection', 'options'),

    Input('update-charts-button', 'n_clicks'),
    Input('dataset-input', 'value'),
    Input('width-input', 'value'),
    Input('saved-data-provider-selection', 'value'),
    Input('start-calculation-button', 'n_clicks'),
    Input('apply-changes-button', 'n_clicks'),

    State('base-instance-id-input', 'value'),
    State('width-input', 'value'),
    State('charts-dimensions-selection', 'value'),
    State('charts-optional-element-selection', 'value'),
    State('charts-data-provider-selection', 'value'),
    *[State(f'feat-{i}', 'value') for i in range(1, len(data_provider.feature_names) + 1)]
)
def combined_callback(
        update_charts_button_n_clicks, dataset_input, width_input, saved_data_provider_selection, start_calculation_button_n_clicks,
        apply_changes_button_n_clicks,
        base_instance_id_input_state, width_input_state, charts_dimensions_selection, charts_optional_element_selection,
        charts_data_provider_selection, *feature_values
):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    selected_tuples = [tuple(map(int, val.strip('()').split(','))) for val in charts_dimensions_selection]
    return_values = [no_update for x in range(10)]
    global data_provider_list
    global data_provider

    needed_figure_data = []
    if charts_data_provider_selection is not None and charts_dimensions_selection is not None:
        for selected_data_provider_index in charts_data_provider_selection:
            temp_data_provider: DataProvider = data_provider_list[selected_data_provider_index]
            temp_data_provider.set_requested_dims(selected_tuples)
            providers_figure_data = temp_data_provider.get_all_figure_data()
            for fig_data in providers_figure_data:
                needed_figure_data.append(fig_data)

    temp = needed_figure_data

    if triggered_id == 'dataset-input':
        data_provider_list = []
        data_provider.set_requested_dims(selected_tuples)
        data_provider.change_dataset(dataset_input)
        data_provider_list.append(data_provider)
        updated_charts = update_chart_grid(update_charts_button_n_clicks, selected_tuples,
                                           charts_optional_element_selection, alt_figure_data=temp)

        dropdown_component = create_information_component(data_provider, data_provider_list)
        chart_grid_component = create_chart_grid_component(data_provider_list)
        top_component = create_settings_component(data_provider, data_provider_list)
        return_values[:6] = updated_charts
        return_values[6] = top_component
        return_values[7] = chart_grid_component
        return_values[8] = dropdown_component

    if triggered_id == 'saved-data-provider-selection':
        data_provider = copy.deepcopy(data_provider_list[saved_data_provider_selection])
        dropdown_component = create_information_component(data_provider, data_provider_list)
        chart_grid_component = create_chart_grid_component(data_provider_list)
        top_component = create_settings_component(data_provider, data_provider_list)

        return_values[6] = top_component
        return_values[7] = chart_grid_component
        return_values[8] = dropdown_component

    if triggered_id == 'update-charts-button':
        data_provider.set_requested_dims(selected_tuples)
        updated_charts = update_chart_grid(update_charts_button_n_clicks, selected_tuples, charts_optional_element_selection, alt_figure_data=temp)
        return_values[:6] = updated_charts if len(updated_charts) <= 6 else updated_charts[:6]

    if triggered_id == 'start-calculation-button':

        new_provider: DataProvider = copy.deepcopy(data_provider)
        new_provider.base_instance_id = base_instance_id_input_state
        new_provider.width = float(width_input_state)
        new_provider.set_requested_dims(selected_tuples)
        new_provider.redo_calculation(dataset_input)
        data_provider_list.append(new_provider)
        data_provider = new_provider
        data_provider.temp_feature_values =  [None for x in range(len(data_provider.base_instance))]

        updated_charts = update_chart_grid(update_charts_button_n_clicks, selected_tuples, charts_optional_element_selection, alt_figure_data=temp)
        return_values[:6] = updated_charts

        dropdown_component = create_information_component(data_provider, data_provider_list)
        chart_grid_component = create_chart_grid_component(data_provider_list)
        top_component = create_settings_component(data_provider, data_provider_list)

        return_values[6] = top_component
        return_values[7] = chart_grid_component
        return_values[8] = dropdown_component

    if triggered_id == 'apply-changes-button':

        options = [{
            'label': ' | '.join(f'{x:.5f}' for x in value.base_instance),
            'value': index} for index, value in enumerate(data_provider_list)]

        if apply_changes_button_n_clicks > 0:
            data_provider.temp_feature_values = feature_values

        return_values[9] = options

    return return_values


@app.callback(
    Output('feature-info-content-output', 'children'),
    Input('feature_info_dropdown', 'value')
)
def change_feature_info(dropdown_value):
    return set_feature_information_tab_content(dropdown_value, data_provider)


if __name__ == '__main__':
    app.run_server(debug=True)