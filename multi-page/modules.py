from typing import Callable

import pandas as pd
from plots import plot_auc_curve, plot_precision_recall_curve, plot_score_distribution

from shiny import Inputs, Outputs, Session, module, render, ui, reactive
from shinywidgets import output_widget, render_widget

import plotly.express as px
import plotly.subplots as sp

import requests
from bs4 import BeautifulSoup
import time
import random
import string
import asyncio
import webbrowser
import os
import paramiko
import stat
import numpy as np


@module.ui
def test_view_ui():
    return ui.nav_panel(
        "Metagenomics",
        ui.layout_columns(
            ui.navset_tab(
                ui.nav_panel("Count Data",
                             ui.layout_sidebar(
                                 ui.sidebar(
                                     ui.accordion(
                                         ui.accordion_panel("Labeling",
                                                            ui.input_radio_buttons("log_or_linear_count", "Linear/Log",
                                                                                   { "linear": "linear", "log": "log",}),
                                                            ui.input_text("plot_title_count", "Plot title",
                                                                          value="Count Data"),
                                                            ui.input_text("x_label_count", "X-axis label",
                                                                          value="Organisms"),
                                                            ui.input_text("y_label_count", "Y-axis label",
                                                                          value="Count"),
                                                            ui.input_slider("figure_width_count", "Figure Witdh", min=300,
                                                                            max=2000, step=50, value=1000),
                                                            ui.input_slider("figure_height_count", "Figure Height", min=300,
                                                                            max=2000, step=50, value=700),
                                                            ui.input_action_button("update_labeling_count", "Update",
                                                                                   class_="btn-success"),

                                                            open=None
                                                            )
                                     ),
                                     ui.input_switch("log_transform_count", "Log Transformation", value=False),
                                     ui.input_select("x_axis_count", "X-axis", choices=['Loading...']),
                                     ui.input_select("taxrank_count", "Taxrank", choices=['Loading...']),
                                     ui.input_selectize("experiment_id_count", "Experiment ID", choices=['Loading...'],
                                                        multiple=True, selected="Loading..."),
                                     ui.input_radio_buttons("combined_or_separate_count", "Combined/Separate",
                                                            {"combined": "combined", "separate": "separate"}),
                                     ui.input_slider("count_count", "Count", min=1, max=1000, step=1,
                                                     value=100),
                                     ui.input_slider("top_selector_count", "Select Top N organisms", min=1,
                                                     max=30, step=1, value=10),
                                     ui.input_select("color_count", "Color", choices=['Loading...']),

                                     ui.input_radio_buttons("mean_sum_count", "Mean/Sum",
                                                            {"mean": "mean", "sum": "sum"}),

                                 ),
                                 output_widget("plot_count"),
                                 ui.output_data_frame("table_count"))

                    ),
                    ui.nav_panel("Percentage Data",
                                 ui.layout_sidebar(
                                        ui.sidebar(
                                            ui.accordion(
                                                ui.accordion_panel("Labeling",
                                                                   ui.input_radio_buttons("log_or_linear_percentage",
                                                                                          "Linear/Log", { "linear": "linear", "log": "log",}),
                                                                   ui.input_text("plot_title_percentage", "Plot title",
                                                                                 value="Percentage Data"),
                                                                   ui.input_text("x_label_percentage", "X-axis label",
                                                                                 value="Organisms"),
                                                                   ui.input_text("y_label_percentage", "Y-axis label",
                                                                                 value="Percentage"),
                                                                   ui.input_slider("figure_width_percentage", "Figure Witdh",
                                                                                   min=300,
                                                                                   max=2000, step=50, value=1000),
                                                                   ui.input_slider("figure_height_percentage",
                                                                                   "Figure Height", min=300,
                                                                                   max=2000, step=50, value=700),
                                                                   ui.input_action_button("update_labeling_percentage",
                                                                                          "Update",
                                                                                          class_="btn-success"),
                                                                   open=None
                                                                   )
                                            ),
                                        ui.input_switch("log_trans_percentage", "Log Transformation", value=False),
                                         ui.input_select("x_axis_percentage", "X-axis", choices=[]),
                                         ui.input_select("taxrank_percentage", "Taxrank", choices=[]),
                                         ui.input_selectize("experiment_id_percentage", "Experiment ID", choices=[], multiple=True),

                                         ui.input_slider("count_percentage", "Count", min=1, max=1000, step=1, value=100),
                                         ui.input_slider("top_selector_percentage", "Select Top N organisms", min=1, max=30, step=1, value=10),
                                         ui.input_select("color_percentage", "Color", choices=[]),

                                         ui.input_radio_buttons("data_scope_percentage", "Data Scope", {"complete": "Complete", "top_n": "Top N"})
                                     ),
                                         output_widget("plot_percentage"),
                                            ui.output_data_frame("table_percentage")
                                 ),

                    ),
                    ui.nav_panel("Total Occurence",
                                 ui.layout_sidebar(
                                     ui.sidebar(
                                         ui.accordion(
                                             ui.accordion_panel("Labeling",
                                                                ui.input_text("plot_title_occurence", "Plot title",
                                                                              value="Gemiddeld Percentage van Totaal Aantal Barcodes"),
                                                                ui.input_text("x_label_occurence", "X-axis label",
                                                                              value="Percentage"),
                                                                ui.input_text("y_label_occurence", "Y-axis label",
                                                                              value="Organisms"),
                                                                ui.input_slider("figure_width_occurence", "Figure Witdh",
                                                                                min=300,
                                                                                max=2000, step=50, value=1000),
                                                                ui.input_slider("figure_height_occurence", "Figure Height",
                                                                                min=300,
                                                                                max=2000, step=50, value=700),
                                                                ui.input_action_button("update_labeling_occurence",
                                                                                       "Update",
                                                                                       class_="btn-success"),
                                                                open=None
                                                                )
                                         ),
                                         ui.input_select("taxrank_occurence", "Taxrank", choices=[]),
                                         ui.input_selectize("experiment_id_occurence", "Experiment ID", choices=[],
                                                            multiple=True),
                                         ui.input_slider("count_occurence", "Count", min=1, max=1000, step=1,
                                                         value=100),
                                         ui.input_slider("top_selector_occurence", "Select Top N organisms", min=1,
                                                         max=30, step=1, value=10),
                                     ),
                                 output_widget("test_plot"),
                                 ui.output_data_frame("table_occurence")
                                 ),
                ),
            ),
        ),
    )


@module.server
def test_view_server(
        input: Inputs, output: Outputs, session: Session, df: reactive.Value
):

    table_dataframe_count = reactive.value(pd.DataFrame())
    table_dataframe_percentage = reactive.value(pd.DataFrame())
    table_dataframe_occurence = reactive.value(pd.DataFrame())
    print("test")

    def save_table_to_csv(dataframe, filename):
        """
        Save a pandas DataFrame to a CSV file.
        """
        dataframe.to_csv(filename, index=False)

    @reactive.effect
    def x_axis():
        print('test')
        if df.get().shape[0] != 0:
            choices = df.get().columns.tolist()
            if "scientific_name" in choices:
                selection = "scientific_name"
            else:
                selection = choices[0]
            ui.update_select("x_axis_count", choices=choices, selected=selection)
            ui.update_select("x_axis_percentage", choices=choices, selected=selection)

    @reactive.effect
    def colors():
        if df.get().shape[0] != 0:
            choices = df.get().columns.tolist()
            if "scientific_name" in choices:
                selection = "scientific_name"
            else:
                selection = choices[0]
            ui.update_select("color_count", choices=choices, selected=selection)
            ui.update_select("color_percentage", choices=choices, selected=selection)
            ui.update_select("color_occurence", choices=choices, selected=selection)

    # @reactive.effect
    # def y_axis():
    #     if df.get().shape[0] != 0:
    #         numeric_columns = df.get().select_dtypes(include=[np.number]).columns.tolist()
    #         if "count" in numeric_columns:
    #             selection = "count"
    #         else:
    #             selection = numeric_columns[0]
    #         ui.update_select("y_axis", choices=numeric_columns, selected=selection)

    @reactive.effect
    def experiment_id():
        if df.get().shape[0] != 0:
            choices = df.get()['experimentid'].unique().tolist()
            ui.update_selectize("experiment_id_count", choices=choices, selected=choices[0])
            ui.update_selectize("experiment_id_percentage", choices=choices, selected=choices[0])
            ui.update_selectize("experiment_id_occurence", choices=choices, selected=choices[0])

    @reactive.effect
    def taxrank():
        if df.get().shape[0] != 0:
            choices = df.get()['taxrank'].unique().tolist()
            if "S" in choices:
                selection = "S"
            ui.update_select("taxrank_count", choices=choices, selected=selection)
            ui.update_select("taxrank_percentage", choices=choices, selected=selection)
            ui.update_select("taxrank_occurence", choices=choices, selected=selection)

    # @reactive.calc
    # @reactive.event(input.x_axis, input.y_axis, input.experiment_id, input.taxrank, input.count)
    # def data_frame_maken():

    @render_widget
    @reactive.event(input.x_axis_count, input.experiment_id_count, input.taxrank_count, input.count_count,
                    input.color_count, input.top_selector_count, input.log_or_linear_count, input.mean_sum_count,
                    input.update_labeling_count, input.figure_width_count, input.figure_height_count,
                    input.combined_or_separate_count,
                    input.log_transform_count)
    def plot_count():
        dataframe = df.get()
        if dataframe.shape[0] != 0 and input.x_axis_count() != "Loading...":
            print(input.combined_or_separate_count())
            if input.combined_or_separate_count() == 'separate':
                print("test")
                filtered_dataframe = dataframe[
                    (dataframe['experimentid'].isin(input.experiment_id_count.get())) &
                    (dataframe['taxrank'] == input.taxrank_count()) &
                    (dataframe['count'] > input.count_count())
                    ].assign(
                    experiment_organism=lambda x: x['experimentid'].astype(str) + " - " + x['scientific_name']
                )

                aggregation_method = input.mean_sum_count()

                # Conditionally apply log transformation
                if input.log_transform_count():
                    filtered_dataframe['count'] = np.log1p(filtered_dataframe['count'])

                filtered_dataframe['aggregated_count'] = filtered_dataframe.groupby('experiment_organism')[
                    'count'].transform(aggregation_method)

                df_top_organisms = filtered_dataframe.sort_values(by='aggregated_count',
                                                                  ascending=False).drop_duplicates(
                    subset='experiment_organism').head(input.top_selector_count())
                print(df_top_organisms)
                fig = px.histogram(
                    data_frame=df_top_organisms,
                    x='experiment_organism',
                    y='aggregated_count',
                    color=input.color_count(),
                    color_discrete_sequence=px.colors.qualitative.Plotly,
                    width=input.figure_width_count(),
                    height=input.figure_height_count()
                ).update_layout(
                    title=input.plot_title_count(),
                    yaxis_type=input.log_or_linear_count(),
                    xaxis_title=input.x_label_count(),
                    yaxis_title=input.y_label_count(),
                )
            if input.combined_or_separate_count() == 'combined':
                print("combined")

                # Filter the dataframe based on the input criteria
                filtered_df = dataframe[
                    (dataframe['experimentid'].isin(input.experiment_id_count.get())) &
                    (dataframe['taxrank'] == input.taxrank_count()) &
                    (dataframe['count'] > input.count_count())
                    ]

                aggregation_method = input.mean_sum_count()

                # Apply log transformation conditionally
                if input.log_transform_count():
                    filtered_df['count'] = np.log1p(filtered_df['count'])

                # Group by 'scientific_name' and aggregate based on the selected method
                df_grouped = filtered_df.groupby('scientific_name').agg({
                    'count': aggregation_method,
                    **{col: 'first' for col in filtered_df.columns if col not in ['scientific_name', 'count']}
                }).reset_index()

                # Get the top N organisms based on the aggregated 'count'
                df_top_organisms = df_grouped.nlargest(input.top_selector_count(), columns='count')

                # Create the histogram using Plotly Express
                fig = px.histogram(
                    data_frame=df_top_organisms,
                    x=input.x_axis_count(),
                    y='count',
                    color=input.color_count(),
                    color_discrete_sequence=px.colors.qualitative.Plotly,
                    width=input.figure_width_count(),
                    height=input.figure_height_count()
                ).update_layout(
                    title=input.plot_title_count(),
                    yaxis_type=input.log_or_linear_count(),
                    xaxis_title=input.x_label_count(),
                    yaxis_title=input.y_label_count(),
                )

                # Save the filtered and aggregated data to CSV
                table_dataframe_count = df_top_organisms
                table_dataframe_count.to_csv("count_data.csv", index=False)

            return fig

    @render.data_frame
    @reactive.event(table_dataframe_count)
    def table_count():
        if table_dataframe_count.get().shape[0] != 0:
            dataframe = table_dataframe_count.get()
            dataframe['count'] = dataframe['count'].astype(int)
            return render.DataGrid(dataframe, row_selection_mode="single", summary=True, )

    @render.data_frame
    @reactive.event(table_dataframe_percentage)
    def table_percentage():
        if table_dataframe_percentage.get().shape[0] != 0:
            dataframe = table_dataframe_percentage.get()
            dataframe['percentage'] = dataframe['percentage'].round()
            return render.DataGrid(dataframe[['scientific_name', 'percentage']] , row_selection_mode="single", summary=True, )

    @render.data_frame
    @reactive.event(table_dataframe_occurence)
    def table_occurence():
        if table_dataframe_occurence.get().shape[0] != 0:
            dataframe = table_dataframe_occurence.get()
            # dataframe['percentage'] = dataframe['percentage'].astype(int)
            return render.DataGrid(dataframe, row_selection_mode="single", summary=True, )

    @render_widget
    @reactive.event(input.x_axis_percentage, input.experiment_id_percentage, input.taxrank_percentage,
                    input.count_percentage,
                    input.color_percentage, input.top_selector_percentage, input.log_or_linear_percentage,
                    input.data_scope_percentage, input.update_labeling_percentage, input.figure_width_percentage,
                    input.figure_height_percentage, input.log_trans_percentage)
    def plot_percentage():
        dataframe = df.get()
        dataframe = dataframe[(dataframe['experimentid'].isin(input.experiment_id_percentage.get())) &
                              (dataframe['taxrank'] == input.taxrank_percentage()) &
                              (dataframe['count'] > input.count_percentage())]

        # Check if log normalization is selected and apply log transformation
        if input.log_trans_percentage():
            dataframe['count'] = np.log1p(dataframe['count'])

        if input.data_scope_percentage() == 'complete':
            total_counts = dataframe['count'].sum()
            print(total_counts, dataframe['count'])
            df_total_counts = dataframe.groupby('scientific_name')['count'].sum().reset_index()
            df_total_counts['percentage'] = (df_total_counts['count'] / total_counts) * 100
        elif input.data_scope_percentage() == 'top_n':
            df_total_counts = dataframe.groupby('scientific_name')['count'].sum().reset_index()
            df_top_organisms = df_total_counts.nlargest(input.top_selector_percentage(), 'count')
            top_total_counts = df_top_organisms['count'].sum()
            df_top_organisms['percentage'] = (df_top_organisms['count'] / top_total_counts) * 100
            df_total_counts = df_top_organisms

        df_top_organisms = df_total_counts.nlargest(input.top_selector_percentage(), 'percentage')
        fig = px.histogram(
            data_frame=df_top_organisms,
            x=input.x_axis_percentage(),
            y='percentage',
            color=input.color_percentage(),
            color_discrete_sequence=px.colors.qualitative.Plotly,
            width=input.figure_width_percentage(),
            height=input.figure_height_percentage()
        ).update_layout(
            title=input.plot_title_percentage(),
            yaxis_type=input.log_or_linear_percentage(),
            xaxis_title=input.x_label_percentage(),
            yaxis_title=input.y_label_percentage()
        )
        table_dataframe_percentage.set(df_top_organisms)
        save_table_to_csv(table_dataframe_percentage.get()[['scientific_name', 'percentage']], "percentage_data.csv")
        return fig

    @render_widget
    @reactive.event(input.x_axis, input.y_axis, input.experiment_id_2, input.taxrank, input.count, input.color,
                    input.top_selector, input.log_or_linear, input.mean_sum, input.data_scope, input.update_labeling_occurence)
    def plot():
        print(input.top_selector())
        if input.x_axis() != None and input.y_axis() != None:
            dataframe = df.get()
            dataframe = dataframe[(dataframe['experimentid'].isin(input.experiment_id_2.get())) &
                                  (dataframe['taxrank'] == input.taxrank()) &
                                  (dataframe['count'] > input.count())]

            if input.y_axis() == 'count':
                aggregation_method = input.mean_sum()
                df_total_counts = dataframe.groupby('scientific_name')['count'].agg(aggregation_method).reset_index()
            elif input.y_axis() == 'percentage':
                ui.update_radio_buttons("log_or_linear", selected="linear")
                # Aggregate total counts for each organism across the selected samples
                if input.data_scope() == 'complete':
                    total_counts = dataframe['count'].sum()
                    df_total_counts = dataframe.groupby('scientific_name')['count'].sum().reset_index()
                    df_total_counts['percentage'] = (df_total_counts['count'] / total_counts) * 100
                elif input.data_scope() == 'top_n':
                    df_total_counts = dataframe.groupby('scientific_name')['count'].sum().reset_index()
                    df_top_organisms = df_total_counts.nlargest(input.top_selector(), columns='count')
                    top_total_counts = df_top_organisms['count'].sum()
                    df_top_organisms['percentage'] = (df_top_organisms['count'] / top_total_counts) * 100
                    df_total_counts = df_top_organisms

            # Filter the original dataframe based on the selected organisms if necessary
            if input.data_scope() == 'complete':
                df_top_organisms = df_total_counts.nlargest(input.top_selector(), columns=input.y_axis())
                dataframe = dataframe[dataframe['scientific_name'].isin(df_top_organisms['scientific_name'])]
            else:
                # already filtered to top N in the percentage case
                if input.y_axis() != 'percentage':
                    df_top_organisms = df_total_counts.nlargest(input.top_selector(), columns=input.y_axis())

            x_axis_selection = input.x_axis()
            y_axis_selection = input.y_axis()
            fig = (px.histogram(
                data_frame=df_top_organisms,
                x=x_axis_selection,
                y=y_axis_selection,
                color=input.color(),
                color_discrete_sequence=px.colors.qualitative.Plotly,
            ).update_layout(
                title=input.plot_title_occurence(),
                yaxis_type=input.log_or_linear_occurence(),
                xaxis_title=input.x_label_occurence(),
                yaxis_title=input.y_label_occurence()
            )
            ).update_xaxes(tickangle=45)
            table_dataframe_occurence.set(df_top_organisms)

            return fig

    @render_widget
    @reactive.event(input.experiment_id_occurence, input.taxrank_occurence, input.count_occurence,
                    input.top_selector_occurence, input.update_labeling_occurence, input.figure_width_occurence, input.figure_height_occurence)
    def test_plot():
        if len(input.experiment_id_occurence()) > 0 and df.get().shape[0] != 0:
            dataframe = df.get()
            dataframe['unique_barcode'] = dataframe['experimentid'].astype(str) + '_' + dataframe['barcode'].astype(str)

            # Filter for multiple experiment IDs
            filtered_df = dataframe[(dataframe['count'] > input.count_occurence()) &
                                    (dataframe['experimentid'].isin(input.experiment_id_occurence.get())) &
                                    (dataframe['taxrank'] == input.taxrank_occurence())]
            # Berekenen van het totale aantal barcodes per experiment
            # total_barcodes_per_experiment = dataframe.groupby('experimentid')['unique_barcode'].nunique()
            total_barcodes = filtered_df['unique_barcode'].nunique()

            # Berekenen hoe vaak elk organisme voorkomt in de verschillende barcodes per experiment
            organism_occurrence_per_experiment = filtered_df.groupby(['scientific_name'])[
                'unique_barcode'].nunique().reset_index(name='barcode_count')
            # Omzetting van de count naar percentages
            organism_occurrence_per_experiment['percentage'] = organism_occurrence_per_experiment.apply(
                lambda row: (row['barcode_count'] / total_barcodes) * 100, axis=1)

            top_10_occurrence_per_experiment = organism_occurrence_per_experiment.groupby('scientific_name').apply(
                lambda x: x.nlargest(input.top_selector_occurence(), 'percentage')).reset_index(drop=True)

            # Gecombineerde plot maken voor alle experimenten met het gemiddelde percentage
            combined_df = top_10_occurrence_per_experiment.groupby('scientific_name')['percentage'].mean().reset_index()
            top_10_combined = combined_df.nlargest(input.top_selector_occurence(), 'percentage')

            fig2 = px.bar(
                top_10_combined,
                x='percentage',
                y='scientific_name',

                color='scientific_name',
                labels={'percentage': 'Gemiddeld Percentage van Totaal Aantal Barcodes',
                        'scientific_name': 'Organisme'},
                color_discrete_sequence=px.colors.qualitative.Plotly,
                width=input.figure_width_occurence(),
                height=input.figure_height_occurence()
            ).update_layout(
                title=input.plot_title_occurence(),
                xaxis_title=input.x_label_occurence(),
                yaxis_title=input.y_label_occurence(),
                xaxis=dict(
                    range=[0, 100]
                ),
            )
            table_dataframe_occurence.set(top_10_combined)
            save_table_to_csv(table_dataframe_occurence.get(), "occurence_data.csv")
            return fig2

    @render.data_frame
    def metagenomics_tabel():
        dataframe = df.get()
        if input.x_axis() != None and input.y_axis() != None:
            dataframe = dataframe[(dataframe['experimentid'] == input.experiment_id_2.get()[0]) & (
                        dataframe['taxrank'] == input.taxrank()) & (dataframe['count'] > input.count())]
            if input.y_axis() in ['count', 'percentage']:
                df_total_counts = dataframe.groupby('scientific_name')['count'].sum().reset_index()

                df_top_organisms = df_total_counts.nlargest(input.top_selector(), columns='count')

                dataframe = dataframe[dataframe['scientific_name'].isin(df_top_organisms['scientific_name'])]
                print(dataframe['percentage'].mean())
            return render.DataGrid(dataframe, row_selection_mode="single", filters=True, summary=True)

    @render.data_frame
    @reactive.event(input.count, input.experiment_id_2, input.taxrank, input.y_axis, input.top_selector)
    def metagenomics_tabel_occurence():
        dataframe = df.get()
        dataframe = dataframe[(dataframe['count'] > input.count()) &
                              (dataframe['experimentid'].isin(input.experiment_id_2.get())) &
                              (dataframe['taxrank'] == input.taxrank())]
        return render.DataGrid(dataframe, row_selection_mode="multiple", filters=True, summary=True)

    # @reactive.effect
    # @reactive.event(input.metagenomics_tabel_selected_rows)
    # def select_all():
    #     input.metagenomics_tabel_occurence_selected_rows.set(1)


@module.ui
def culturomics_view_ui():
    return ui.nav_panel(
        "Culturomics",
        ui.layout_sidebar(
            ui.sidebar(
                ui.card_header("Edit Data"),
                ui.input_select("culturomics_x_axis", "X-axis", choices=[]),
                ui.input_select("culturomics_y_axis", "Y-axis", choices=[]),
                ui.input_slider("culturomics_count", "Count", min=1, max=1000, step=1, value=100),
                ui.input_slider("culturomics_top_selector", "Select Top N organisms", min=1, max=30, step=1, value=10),
                ui.input_select("culturomics_color", "Color", choices=[]),
            ),
            ui.output_data_frame("culturomics_data"),
            ui.output_data_frame("metagenomics_data")
        ),
    )


@module.server
def culturomics_view_server(
        input: Inputs, output: Outputs, session: Session, df_culturomics: reactive.Value,
        df_metagenomics: reactive.Value
):
    @render.data_frame
    def culturomics_data():
        dataframe = df_culturomics.get()
        return render.DataGrid(dataframe, row_selection_mode="single", filters=True, summary=True)

    @render.data_frame
    @reactive.event(input.culturomics_data_selected_rows, input.culturomics_count)
    def metagenomics_data():
        if input.culturomics_data_selected_rows():
            metagenomics_dataframe = df_metagenomics.get()
            culturomics_dataframe = df_culturomics.get()
            selected_row = input.culturomics_data_selected_rows()
            selected_organism = " ".join(culturomics_dataframe.loc[selected_row[0]]['wgs_organisme'].split(' ')[0:2])
            metagenomics_dataframe = metagenomics_dataframe[
                (metagenomics_dataframe['scientific_name'] == selected_organism) & (
                            metagenomics_dataframe['count'] > input.culturomics_count())]

            return render.DataGrid(metagenomics_dataframe)

    @reactive.effect
    def culturomics_x_axis():
        if df_culturomics.get().shape[0] != 0:
            choices = df_culturomics.get().columns.tolist()
            if "scientific_name" in choices:
                selection = "scientific_name"
            else:
                selection = choices[0]
            ui.update_select("culturomics_x_axis", choices=choices, selected=selection)

    @reactive.effect
    def culturomics_colors():
        if df_culturomics.get().shape[0] != 0:
            choices = df_culturomics.get().columns.tolist()
            if "scientific_name" in choices:
                selection = "scientific_name"
            else:
                selection = choices[0]
            ui.update_select("culturomics_color", choices=choices, selected=selection)

    @reactive.effect
    def culturomics_y_axis():
        if df_culturomics.get().shape[0] != 0:
            numeric_columns = df_culturomics.get().select_dtypes(include=[np.number]).columns.tolist()
            if "count" in numeric_columns:
                selection = "count"
            else:
                selection = numeric_columns[0]
            ui.update_select("culturomics_y_axis", choices=numeric_columns, selected=selection)

    @render_widget
    @reactive.event(input.x_axis, input.y_axis, input.experiment_id_2, input.taxrank, input.count)
    def test_plot():
        if input.x_axis() != None and input.y_axis() != None and len(input.experiment_id_2()) > 0 and df.get().shape[
            0] != 0:
            dataframe = df.get()
            dataframe['unique_barcode'] = dataframe['experimentid'].astype(str) + '_' + dataframe['barcode'].astype(str)

            # Filter for multiple experiment IDs
            filtered_df = dataframe[(dataframe['count'] > input.count()) &
                                    (dataframe['experimentid'].isin(input.experiment_id_2.get())) &
                                    (dataframe['taxrank'] == input.taxrank())]
            # Berekenen van het totale aantal barcodes per experiment
            # total_barcodes_per_experiment = dataframe.groupby('experimentid')['unique_barcode'].nunique()
            total_barcodes = filtered_df['unique_barcode'].nunique()

            # Berekenen hoe vaak elk organisme voorkomt in de verschillende barcodes per experiment
            organism_occurrence_per_experiment = filtered_df.groupby(['scientific_name'])[
                'unique_barcode'].nunique().reset_index(name='barcode_count')
            # Omzetting van de count naar percentages
            organism_occurrence_per_experiment['percentage'] = organism_occurrence_per_experiment.apply(
                lambda row: (row['barcode_count'] / total_barcodes) * 100, axis=1)

            # Vinden van de top 20 organismen per experiment
            top_10_occurrence_per_experiment = organism_occurrence_per_experiment.groupby('scientific_name').apply(
                lambda x: x.nlargest(20, 'percentage')).reset_index(drop=True)

            # Gecombineerde plot maken voor alle experimenten met het gemiddelde percentage
            combined_df = top_10_occurrence_per_experiment.groupby('scientific_name')['percentage'].mean().reset_index()
            top_10_combined = combined_df.nlargest(20, 'percentage')

            fig2 = px.bar(
                top_10_combined,
                x='percentage',
                y='scientific_name',

                color='scientific_name',
                labels={'percentage': 'Gemiddeld Percentage van Totaal Aantal Barcodes',
                        'scientific_name': 'Organisme'},
            )

            fig2.update_layout(
                xaxis_title="Gemiddeld Percentage van Totaal Aantal Barcodes",
                yaxis_title="Organisme",
                title=f"Gemiddeld Percentage Voorkomen van Top 20 Organismen over experimenten: {' '.join(input.experiment_id_2.get())}",
                xaxis=dict(
                    range=[0, 100]
                )
            )

            return fig2


@module.ui
def test2_view_ui():
    return ui.nav_panel(
        "Test2",
        ui.layout_columns(
            ui.value_box(
                title="Row count",
                value=ui.output_text("row_count"),
                theme="primary",
            ),
            ui.value_box(
                title="Rows selected",
                value=ui.output_ui("rows"),
                theme="bg-green",
            ),
            gap="20px",
        ),
        ui.layout_columns(
            ui.card(ui.output_data_frame("data")),
            style="margin-top: 20px;",
        ),
        ui.card(  # Add a new card
            ui.card_header("Edit Data"),
            ui.input_select("table_selector", "Select Table", choices=[]),
            ui.input_select("experiment_selector", "Select experiment", choices=[]),
            ui.input_action_button("update", "Update", class_="btn-success")
        )

    )


@module.server
def test2_view_server(
        input: Inputs, output: Outputs, session: Session, db_manager: Callable
):
    flag = reactive.value(0)
    df = reactive.value(pd.DataFrame())
    @reactive.effect
    def df_table_aanmaken():
        if input.table_selector() != None:
            selected_table = input.table_selector()

            try:
                db_manager.execute(f"SELECT * FROM {selected_table}")
                result = db_manager.fetch_all()
                columns = db_manager.cursor.description
                df.set(pd.DataFrame(result, columns=[column[0] for column in columns]))
            except Exception as e:
                # Handle database errors here (display an error message, etc.)
                print(f"Error loading data: {e}")

    @render.text
    def row_count():
        return df.get().shape[0]

    @render.text
    def mean_score():
        pass

    @reactive.effect
    def all_tables():
        db_manager.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
        result = db_manager.fetch_all()
        tables = [table_name[0] for table_name in result]
        ui.update_select("table_selector", choices=tables)

    @reactive.effect
    def experiment_selector():
        if df.get().shape[0] != 0:
            if "experimentid" in df.get().columns:
                choices = df.get()['experimentid'].unique().tolist()
                ui.update_select("experiment_selector", choices=choices)

    @render.data_frame
    def data():
        dataframe = df.get()
        if input.experiment_selector():
            dataframe = dataframe[dataframe['experimentid'] == input.experiment_selector()]
        return render.DataGrid(dataframe, row_selection_mode="multiple", filters=True, summary=True)

    @render.ui()
    def rows():
        rows = input.data_selected_rows()
        selected = ", ".join(str(i) for i in sorted(rows)) if rows else "None"
        return f"{selected}"

    @reactive.effect
    def column():
        choices = df.get().columns.tolist()
        ui.update_select("column", choices=choices)


@module.ui
def dotplot_view_ui():
    return ui.nav_panel(
        "Dotplot",
        ui.layout_columns(
            ui.layout_sidebar(
                ui.sidebar(
                    ui.output_text_verbatim("first_selected_organism"),
                    ui.output_text_verbatim("second_selected_organism"),
                    ui.layout_columns(
                        ui.input_action_button("send", "Maak dotplot", class_="btn-success", width="200px",
                                               height="40px", ),
                    ),
                    ui.layout_columns(
                        ui.input_select("jobs", "Select job", choices=['None']),
                    ),
                    ui.input_action_button("goJob", "Go to job", class_="btn-success", width="200px", height="40px",
                                           style="margin-top: 20px;"),
                ),
                ui.layout_columns(
                    ui.output_data_frame("data"),
                    style="margin-top: 20px;",
                ),
                ui.layout_columns(
                    ui.div(
                        ui.input_action_button("select", "Select", class_="btn-success", width="200px",
                                               height="40px"),
                        ui.input_action_button("reset", "Reset", class_="btn-danger", width="200px", height="40px"),
                        style="display: flex; justify-content: center; gap: 20px;"
                    ),
                ),

            )
        )
    )


@module.server
def dotplot_view_server(
        input: Inputs, output: Outputs, session: Session, df: reactive.Value
):
    selected_genus_name = reactive.value(None)
    selection_df = reactive.value(pd.DataFrame())
    first_sample_name = reactive.value(None)
    second_sample_name = reactive.value(None)
    progress = reactive.value(None)
    active_jobs = reactive.value([])
    first_contig_flag = reactive.value(False)
    second_contig_flag = reactive.value(False)

    def generate_id():
        random_part = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
        timestamp_part = time.strftime("%Y%m%d%H%M%S")
        return f"{random_part}_{timestamp_part}"

    def generate_random_email():
        # Generates a random email
        name = ''.join(random.choice(string.ascii_letters) for _ in range(5))
        domain = ''.join(random.choice(string.ascii_letters) for _ in range(5))
        return f"{name}@{domain}.com"

    @render.text
    def row_count():
        return df.get().shape[0]

    @render.text
    def first_selected_organism():
        return f"First sample: {first_sample_name.get()}"

    @render.text
    def second_selected_organism():
        return f"Second sample: {second_sample_name.get()}"

    @reactive.effect
    @reactive.event(input.select)
    def select_sample():
        if input.data_selected_rows():
            if first_sample_name.get() == None:
                first_sample_name.set(df.get().loc[input.data_selected_rows()[0]]['csampleid'])
                selected_genus_name.set(df.get().loc[input.data_selected_rows()[0]]['wgs_organisme'].split(' ')[0])
                if df.get().loc[input.data_selected_rows()[0]]['whole_assembly'] == 'true':
                    first_contig_flag.set(False)
                else:
                    first_contig_flag.set(df.get().loc[input.data_selected_rows()[0]]['whole_assembly'])
            else:
                second_sample_name.set(selection_df.get().loc[input.data_selected_rows()[0]]['csampleid'])
                if selection_df.get().loc[input.data_selected_rows()[0]]['whole_assembly'] == 'true':
                    second_contig_flag.set(False)
                else:
                    second_contig_flag.set(selection_df.get().loc[input.data_selected_rows()[0]]['whole_assembly'])

            # reactive.value(selected_organism)

    @reactive.effect
    @reactive.event(input.reset)
    def reset_sample():
        first_sample_name.set(None)
        second_sample_name.set(None)
        selected_genus_name.set(None)

    @reactive.effect
    @reactive.event(input.send)
    def request_dgenies():
        if first_sample_name.get() != None and second_sample_name.get() != None:
            try:
                ui.notification_show("Downloading first sample", duration=None)
                bestanden_ophalen(first_sample_name.get())
                ui.notification_show("Downloading second sample", duration=None)
                bestanden_ophalen(second_sample_name.get())
            except Exception as e:
                ui.notification_show("Bestand bestaat niet", type='error', duration=None)
                return

            new_id = generate_id()
            email = generate_random_email()
            # Maak een sessie object
            session = requests.Session()

            main_url = "https://dgenies.toulouse.inra.fr/"

            user_agents = [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:88.0) Gecko/20100101 Firefox/88.0",
                "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.110 Mobile Safari/537.36"
            ]

            # Stel algemene headers in voor de sessie
            session.headers.update({
                "accept": "*/*",
                "accept-language": "en-US,en;q=0.9",
                "user-agent": random.choice(user_agents),
                "x-requested-with": "XMLHttpRequest"
            })

            # Initial URL ophalen en s_id extraheren
            initial_url = "https://dgenies.toulouse.inra.fr/run"
            initial_response = session.get(initial_url)
            soup = BeautifulSoup(initial_response.text, 'html.parser')
            body_tag = soup.find('body', onload=True)
            js_code = body_tag['onload']
            s_id = js_code.split('dgenies.run.init')[1].strip()[1:-2].split(',')[0].strip("'")

            # Verstuur een verzoek naar ask-upload
            ask_upload_url = "https://dgenies.toulouse.inra.fr/ask-upload"
            data = {"s_id": s_id}
            ask_upload_response = session.post(ask_upload_url, data=data)
            print(ask_upload_response.text)

            # Upload bestand
            url = "https://dgenies.toulouse.inra.fr/upload"
            formats_value = "fasta"
            file_path_1 = first_sample_name.get() + ".fasta"
            file_path_2 = second_sample_name.get() + ".fasta"
            ui.notification_show("Uploading first sample", duration=None)
            # Gebruik de context manager 'with' om ervoor te zorgen dat het bestand correct wordt gesloten

            if first_contig_flag.get() is not False:
                contig_extracter(file_path_1, first_contig_flag.get())  # Get extracted contig
                with open(f"{first_contig_flag.get()}.fasta", "rb") as f:
                    files = {
                        "s_id": (None, s_id),
                        "formats": (None, formats_value),
                        "file-target": ("file1.fasta", f, "application/octet-stream")
                        # Use extracted contig
                    }
                    response = session.post(url, files=files)
            else:
                with open(file_path_1, "rb") as f:
                    files = {
                        "s_id": (None, s_id),
                        "formats": (None, formats_value),
                        "file-target": ("file1.fasta", f, "application/octet-stream")
                    }
                    response = session.post(url, files=files)

            print(response.status_code, response.text)
            ui.notification_show("Uploading second sample", duration=None)
            # Gebruik de context manager 'with' om ervoor te zorgen dat het bestand correct wordt gesloten
            if second_contig_flag.get() is not False:
                contig_extracter(file_path_2, second_contig_flag.get())  # Get extracted contig
                with open(f"{second_contig_flag.get()}.fasta", "rb") as f:
                    files = {
                        "s_id": (None, s_id),
                        "formats": (None, formats_value),
                        "file-target": ("file2.fasta", f, "application/octet-stream")
                        # Use extracted contig
                    }
                    response = session.post(url, files=files)
            else:
                with open(file_path_2, "rb") as f:
                    files = {
                        "s_id": (None, s_id),
                        "formats": (None, formats_value),
                        "file-target": ("file2.fasta", f, "application/octet-stream")
                    }
                    response = session.post(url, files=files)

            print(response.status_code, response.text)

            os.remove(file_path_1)
            os.remove(file_path_2)

            # Verstuur de laatste analyse-aanvraag
            data = {
                "id_job": new_id,
                "email": "qridderpl@gmail.com",
                "s_id": s_id,
                "type": "align",
                "jobs[0][id_job]": new_id,
                "jobs[0][email]": email,
                "jobs[0][s_id]": s_id,
                "jobs[0][type]": "align",
                "jobs[0][query]": "file1.fasta",
                "jobs[0][query_type]": "local",
                "jobs[0][target]": "file2.fasta",
                "jobs[0][target_type]": "local",
                "jobs[0][tool]": "minimap2",
                "jobs[0][tool_options][]": "repeat:many",
                "nb_jobs": "1",
            }
            response = session.post("https://dgenies.toulouse.inra.fr/launch_analysis", data=data)

            print(response.status_code, response.text)

            redirect_data = response.json()  # Convert response text to a Python dictionary
            redirect_url = redirect_data.get("redirect")
            base_url = "https://dgenies.toulouse.inra.fr"
            full_redirect_url = base_url + '/status/' + redirect_url.split("/")[-1]

            href = None
            progress.set('test')
            while href is None:
                try:
                    progression = f"Loading job: {redirect_url.split('/')[-1]}..."
                    ui.notification_show(progression, duration=None)
                    response = session.get(full_redirect_url)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    status = soup.find('div', class_='status-body')
                    href = status.find('a', href=True)
                    print(progression)
                    time.sleep(5)
                except Exception as e:
                    print(e)
                    time.sleep(5)

            print(base_url + href['href'])
            list_active = active_jobs.get()
            list_active.append(new_id)
            active_jobs.set(list_active)
            ui.update_select("jobs", choices=active_jobs.get(), selected=new_id)
            first_sample_name.set(None)
            second_sample_name.set(None)
            selected_genus_name.set(None)

    def contig_extracter(file_path, contig):
        """Extracts a contig from a FASTA file and writes it to an output file."""
        with open(file_path, 'r') as file, open(f"{contig}.fasta", 'w') as output_file:
            writing_contig = False

            for line in file:  # We only iterate over the input file here
                if line.startswith('>') and writing_contig:
                    break

                if contig in line:
                    writing_contig = True

                if writing_contig:
                    output_file.write(line)

    @render.data_frame
    def data():
        if selected_genus_name.get() != None:
            selection_df.set(df.get()[df.get()['wgs_organisme'].str.contains(selected_genus_name.get())])
            selection_df.get().reset_index(drop=True, inplace=True)
            return render.DataGrid(selection_df.get(), row_selection_mode="single", filters=True, summary=True)
        else:
            dataframe = df.get()
            return render.DataGrid(dataframe, row_selection_mode="single", filters=True, summary=True)

    @render.ui()
    def rows():
        rows = input.data_selected_rows()
        selected = ", ".join(str(i) for i in sorted(rows)) if rows else "None"
        return f"{selected}"

    @reactive.effect
    def column():
        choices = df.get().columns.tolist()
        ui.update_select("column", choices=choices)

    @reactive.effect
    def update_job_list():
        if active_jobs.get() != []:
            ui.update_select("jobs", choices=active_jobs.get())

    @reactive.effect
    @reactive.event(input.goJob)
    def go_job():
        if input.jobs() != 'None':
            base_url = "https://dgenies.toulouse.inra.fr"
            url = base_url + "/result/" + input.jobs()
            webbrowser.open(url)

    def bestanden_ophalen(sample_id):
        HOSTNAME = "145.97.18.149"
        USERNAME = "ridderplaat.q"
        # You'll likely need your SSH password here
        PASSWORD = "Quinten4321or1234!"

        remote_path = "/exports/nas/ridderplaat.q/Culturomics/results2122_v2/CLF2122_1032/assembly/assembly.fasta"
        local_path = os.path.join(".", f"{sample_id}.fasta")  # Download to the current directory
        base_path = "/exports/nas/ridderplaat.q/Culturomics"

        # Create an SSH connection
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(HOSTNAME, username=USERNAME, password=PASSWORD)

        # Create an SFTP client for file transfer
        sftp_client = ssh_client.open_sftp()

        items = sftp_client.listdir_attr(base_path)
        folders = [item.filename for item in items if stat.S_ISDIR(item.st_mode)]
        for folder in folders:
            sub_items = sftp_client.listdir_attr(base_path + "/" + folder)
            sub_folders = [item.filename for item in sub_items if stat.S_ISDIR(item.st_mode)]
            if sample_id in sub_folders:
                remote_path = base_path + "/" + folder + "/" + sample_id + "/assembly/assembly.fasta"
                break
        # Download the file
        sftp_client.get(remote_path, local_path)

        # Close the connections
        sftp_client.close()
        ssh_client.close()
        print("Download complete!")


