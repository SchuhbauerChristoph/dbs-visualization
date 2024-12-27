import dash
from dash import html
from dash.dependencies import Input, Output, State
from dash_app.components.chart_grid_component import create_chart_grid_component, createAllCharts
from dash_app.components.tab_component import create_tabs_component, set_feature_information_tab_content
from dash import no_update, dcc
from dash_app.components.top_component import create_top_component
from dataProvider import DataProvider
import plotly.graph_objects as go
from typing import List
import copy

data_provider = DataProvider()
data_provider_list = [data_provider]
data_provider = data_provider_list[0]
app = dash.Dash(__name__)

app.layout = html.Div(
    children=[
        html.Div(id='dummy-output', style={'display': 'none'}, children=[
            dcc.Store(id='selected-instance', data=None),
            dcc.Store(id='highlight_store', data=None),
            dcc.Store(id='dataset_store', data=None),
            html.Div(id='info_grid_container-6', children=[], style={'display': 'none'}),
            html.Div(id='tabs-example', children=[], style={'display': 'none'})
        ]),

        html.Div(id='hidden-div', style={'display': 'none'}),
        html.Div(
            style={
                'width': '100%',
                'height': '15%',
                'border': '1px solid black',
            },
            id='top-area',
            children=create_top_component(data_provider, data_provider_list)
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
                            children=create_chart_grid_component(data_provider, data_provider_list)
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
                            children=create_tabs_component(data_provider, data_provider_list)
                ),
            ]
        ),
    ],
    style={'height': '100vh', 'display': 'flex', 'flex-direction': 'column'}
)


def get_highlighted_grid_charts(hover_data, figure, heatmap_select) -> List[go.Figure]:
    if hover_data['points'][0]['curveNumber'] == 1:
        instance_data = figure['data'][1]['customdata'][0]['instance_data']

        hovered_point_index = hover_data['points'][0]['pointNumber']
        highlight_instance = instance_data[hovered_point_index]
        figure_data = data_provider.get_all_figure_data()
        chart_figures, colors = createAllCharts(figure_data, highlight_instance=highlight_instance, heatmap_select=heatmap_select)
        return chart_figures
    return [no_update for x in range(6)]


def get_instance_on_hover(
        hover_data_1, hover_data_2, hover_data_3, hover_data_4, hover_data_5, hover_data_6,
        figure_1, figure_2, figure_3, figure_4, figure_5, figure_6, heatmap_select
) -> List[go.Figure]:

    if hover_data_1:
        return get_highlighted_grid_charts(hover_data_1, figure_1, heatmap_select)
    if hover_data_2:
        return get_highlighted_grid_charts(hover_data_2, figure_2, heatmap_select)
    if hover_data_3:
        return get_highlighted_grid_charts(hover_data_3, figure_3, heatmap_select)
    if hover_data_4:
        return get_highlighted_grid_charts(hover_data_4, figure_4, heatmap_select)
    if hover_data_5:
        return get_highlighted_grid_charts(hover_data_5, figure_5, heatmap_select)
    if hover_data_6:
        return get_highlighted_grid_charts(hover_data_6, figure_6, heatmap_select)
    return [no_update for x in range(6)]


def update_chart_grid(n_click, selected_values, heatmap_select, alt_figure_data=None) -> List[go.Figure]:
    if n_click is None:
        return [no_update for i in range(6)]

    if selected_values:
        data_provider.setRequestedDims(selected_values)
    else:
        return [no_update for i in range(6)]

    figure_data = data_provider.get_all_figure_data()
    if alt_figure_data is not None and len(alt_figure_data) != 0:
        print(len(alt_figure_data))
        figure_data = alt_figure_data
    updated_charts, colors = createAllCharts(figure_data, heatmap_select=heatmap_select)

    return updated_charts








@app.callback(
    *[Output(f'chart-{i+1}', 'figure') for i in range(6)],
    Output('top-area', 'children'),
    Output('chart-grid-area', 'children'),
    Output('tabs-area', 'children'),

    *[Input(f'chart-{i + 1}', 'hoverData') for i in range(6)],
    Input('update-charts-button', 'n_clicks'),
    Input('dataset-input', 'value'),
    Input('width-input', 'value'),
    Input('saved-data-provider-selection', 'value'),
    Input('start-calculation-button', 'n_clicks'),

    *[State(f'chart-{i + 1}', 'figure') for i in range(6)],
    State('base-instance-id-input', 'value'),  # instance
    State('width-input', 'value'),  # width
    State('charts-dimensions-selection', 'value'),  # dims
    State('charts-optional-element-selection', 'value'),
    State('charts-data-provider-selection', 'value'),
)
def update_visuals(
        hover_data_1, hover_data_2, hover_data_3, hover_data_4, hover_data_5, hover_data_6,
        update_dimensions_button_n_clicks, dataset_input, width_input, saved_data_provider_selection, start_calculation_button_n_clicks,
        figure_1, figure_2, figure_3, figure_4, figure_5, figure_6,
        base_instance_id_input_state, width_input_state, charts_dimensions_selection, charts_optional_element_selection, charts_data_provider_selection
):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    selected_tuples = [tuple(map(int, val.strip('()').split(','))) for val in charts_dimensions_selection]
    return_values = [no_update for x in range(9)]
    global data_provider_list
    global data_provider

    needed_figure_data = []
    if charts_data_provider_selection is not None and charts_dimensions_selection is not None:
        for selected_data_provider_index in charts_data_provider_selection:
            temp_data_provider: DataProvider = data_provider_list[selected_data_provider_index]
            temp_data_provider.setRequestedDims(selected_tuples)
            providers_figure_data = temp_data_provider.get_all_figure_data()
            for fig_data in providers_figure_data:
                needed_figure_data.append(fig_data)

    temp = needed_figure_data
    hover_data = [hover_data_1, hover_data_2, hover_data_3, hover_data_4, hover_data_5, hover_data_6]
    figures = [figure_1, figure_2, figure_3, figure_4, figure_5, figure_6]

    def handle_hover(hover_data_idx):
        charts = get_instance_on_hover(
            *(hover_data[idx] if idx == hover_data_idx else None for idx in range(6)),
            *figures, charts_optional_element_selection
        )
        return charts

    if triggered_id.startswith('chart-'):
        chart_index = int(triggered_id.split('-')[1]) - 1
        return_values[:6] = handle_hover(chart_index)

    if triggered_id == 'dataset-input':

        data_provider_list = []
        data_provider.setRequestedDims(selected_tuples)
        data_provider.change_dataset(dataset_input)
        data_provider_list.append(data_provider)
        updated_charts = update_chart_grid(update_dimensions_button_n_clicks, selected_tuples,
                                           charts_optional_element_selection, alt_figure_data=temp)

        dropdown_component = create_tabs_component(data_provider, data_provider_list)
        chart_grid_component = create_chart_grid_component(data_provider, data_provider_list)
        top_component = create_top_component(data_provider, data_provider_list)
        return_values[:6] = updated_charts
        return_values[6] = top_component
        return_values[7] = chart_grid_component
        return_values[8] = dropdown_component

    if triggered_id == 'saved-data-provider-selection':
        data_provider = copy.deepcopy(data_provider_list[saved_data_provider_selection])
        dropdown_component = create_tabs_component(data_provider, data_provider_list)
        chart_grid_component = create_chart_grid_component(data_provider, data_provider_list)
        top_component = create_top_component(data_provider, data_provider_list)

        return_values[6] = top_component
        return_values[7] = chart_grid_component
        return_values[8] = dropdown_component

    if triggered_id == 'update-charts-button':
        data_provider.setRequestedDims(selected_tuples)
        updated_charts = update_chart_grid(update_dimensions_button_n_clicks, selected_tuples, charts_optional_element_selection, alt_figure_data=temp)
        return_values[:6] = updated_charts

    if triggered_id == 'start-calculation-button':
        old_provider = copy.deepcopy(data_provider)
        data_provider_list.append(old_provider)

        data_provider.base_instance_id = base_instance_id_input_state
        data_provider.width = float(width_input_state)
        data_provider.setRequestedDims(selected_tuples)
        #data_provider.setRequestedDims([])
        data_provider.redoCalculation(dataset_input)

        updated_charts = update_chart_grid(update_dimensions_button_n_clicks, selected_tuples, charts_optional_element_selection, alt_figure_data=temp)
        return_values[:6] = updated_charts

        dropdown_component = create_tabs_component(data_provider, data_provider_list)
        chart_grid_component = create_chart_grid_component(data_provider, data_provider_list)
        top_component = create_top_component(data_provider, data_provider_list)

        return_values[6] = top_component
        return_values[7] = chart_grid_component
        return_values[8] = dropdown_component

    return return_values


@app.callback(
    #Output('apply-changes-button', 'children'),
    Output('saved-data-provider-selection', 'options'),
    Input('apply-changes-button', 'n_clicks'),
    *[State(f'feat-{i}', 'value') for i in range(1, len(data_provider.feature_names) + 1)]
)
def apply_instance_changes(n_clicks, *feature_values):
    options = [{
                   'label': ' | '.join(f'{x:.5f}' for x in value.base_instance),
                   'value': index} for index, value in enumerate(data_provider_list)]

    if n_clicks == 0:
        #return ' | '.join(f'{x:.5f}' for x in feature_values), options
        return options
    old_provider = copy.deepcopy(data_provider)
    data_provider_list.append(old_provider)
    data_provider.temp_feature_values = feature_values
    return options
    #return ' | '.join(f'{x:.5f}' for x in feature_values), options


@app.callback(
    Output('feature-info-content-output', 'children'),
    Input('feature_info_dropdown', 'value')
)
def change_feature_info(dropdown_value):
    return set_feature_information_tab_content(dropdown_value, data_provider)


if __name__ == '__main__':
    app.run_server(debug=True)