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
            ui.card(  # Add a new card
                ui.card_header("Edit Data"),
                    ui.input_select("x_axis", "X-axis", choices=[]),
                    ui.input_select("y_axis", "Y-axis", choices=[]),
                    ui.input_selectize("experiment_id_2", "Experiment ID", choices=[], multiple=True),
                    ui.input_select("taxrank", "Tax Rank", choices=[]),
                    ui.input_slider("count", "Count", min=1, max=1000, step=1, value=100),
                    ui.input_slider("top_selector", "Select Top N organisms", min=1, max=30, step=1, value=10),
                    ui.input_select("color", "Color", choices=[]),
                ),

            ui.card(
        output_widget("plot"),
        output_widget("test_plot"),
        ),
        col_widths=(2, 10)
        ),
        )



@module.server
def test_view_server(
    input: Inputs, output: Outputs, session: Session, df: reactive.Value
):

    @reactive.effect
    def x_axis():
        if df.get().shape[0] != 0:
            choices = df.get().columns.tolist()
            if "scientific_name" in choices:
                selection = "scientific_name"
            else:
                selection = choices[0]
            ui.update_select("x_axis", choices=choices, selected=selection)

    @reactive.effect
    def colors():
        if df.get().shape[0] != 0:
            choices = df.get().columns.tolist()
            if "scientific_name" in choices:
                selection = "scientific_name"
            else:
                selection = choices[0]
            ui.update_select("color", choices=choices, selected=selection)

    @reactive.effect
    def y_axis():
        if df.get().shape[0] != 0:
            numeric_columns = df.get().select_dtypes(include=[np.number]).columns.tolist()
            if "count" in numeric_columns:
                selection = "count"
            else:
                selection = numeric_columns[0]
            ui.update_select("y_axis", choices=numeric_columns, selected=selection)

    @reactive.effect
    def experiment_id():
        if df.get().shape[0] != 0:
            choices = df.get()['experimentid'].unique().tolist()
            ui.update_selectize("experiment_id_2", choices=choices, selected=choices[0])

    @reactive.effect
    def taxrank():
        if df.get().shape[0] != 0:
            choices = df.get()['taxrank'].unique().tolist()
            if "S" in choices:
                selection = "S"
            ui.update_select("taxrank", choices=choices, selected=selection)

    # @reactive.calc
    # @reactive.event(input.x_axis, input.y_axis, input.experiment_id, input.taxrank, input.count)
    # def data_frame_maken():






    @render_widget
    @reactive.event(input.x_axis, input.y_axis, input.experiment_id_2, input.taxrank, input.count, input.color, input.top_selector)
    def plot():
        print(input.top_selector())
        if input.x_axis() != None and input.y_axis() != None:
            dataframe = df.get()
            dataframe = dataframe[(dataframe['experimentid'] == input.experiment_id_2.get()[0]) & (dataframe['taxrank'] == input.taxrank()) & (dataframe['count'] > input.count())]
            if input.y_axis() == 'count':
                # Calculate total count per organism
                df_total_counts = dataframe.groupby('scientific_name')['count'].sum().reset_index()

                # Select top n organisms by total count
                df_top_organisms = df_total_counts.nlargest(input.top_selector(), columns='count')

                # Filter the original dataframe based on the selected organisms
                dataframe = dataframe[dataframe['scientific_name'].isin(df_top_organisms['scientific_name'])]

            x_axis_selection = input.x_axis()
            y_axis_selection = input.y_axis()
            fig = (px.histogram(
                data_frame=dataframe,
                x=x_axis_selection,
                y=y_axis_selection,
                color=input.color(),
                color_discrete_sequence=px.colors.qualitative.Plotly,
            )
            .update_layout(
                title="Test Plot",
                yaxis_type='log'
            ).update_xaxes(tickangle=45)
            ).update_yaxes(showgrid=True, minor_showgrid=False)
            return fig
    @render_widget
    @reactive.event(input.x_axis, input.y_axis, input.experiment_id_2, input.taxrank, input.count, input.top_selector)
    def test_plot():
        if input.x_axis() != None and input.y_axis() != None and len(input.experiment_id_2()) > 0 and df.get().shape[0] != 0:
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

            top_10_occurrence_per_experiment = organism_occurrence_per_experiment.groupby('scientific_name').apply(
                lambda x: x.nlargest(input.top_selector(), 'percentage')).reset_index(drop=True)

            # Gecombineerde plot maken voor alle experimenten met het gemiddelde percentage
            combined_df = top_10_occurrence_per_experiment.groupby('scientific_name')['percentage'].mean().reset_index()
            top_10_combined = combined_df.nlargest(input.top_selector(), 'percentage')

            fig2 = px.bar(
                top_10_combined,
                x='percentage',
                y='scientific_name',

                color='scientific_name',
                labels={'percentage': 'Gemiddeld Percentage van Totaal Aantal Barcodes',
                        'scientific_name': 'Organisme'},
                color_discrete_sequence=px.colors.qualitative.Plotly,
            )

            fig2.update_layout(
                xaxis_title="Gemiddeld Percentage van Totaal Aantal Barcodes",
                yaxis_title="Organisme",
                title=f"Gemiddeld Percentage Voorkomen van Top 20 Organismen over experimenten: {', '.join(input.experiment_id_2.get())}",
                xaxis=dict(
                    range=[0, 100]
                )
            )

            return fig2

@module.ui
def culturomics_view_ui():
    return ui.nav_panel(
        "Culturomics",
        ui.layout_columns(
            ui.card(  # Add a new card
                ui.card_header("Edit Data"),
                    ui.input_select("culturomics_x_axis", "X-axis", choices=[]),
                    ui.input_select("culturomics_y_axis", "Y-axis", choices=[]),
                    ui.input_slider("culturomics_count", "Count", min=1, max=1000, step=1, value=100),
                    ui.input_slider("culturomics_top_selector", "Select Top N organisms", min=1, max=30, step=1, value=10),
                    ui.input_select("culturomics_color", "Color", choices=[]),
                ),

            ui.card(
                    ui.output_data_frame("culturomics_data"),
                    ui.output_data_frame("metagenomics_data")
        ),
        col_widths=(2, 10)
        ),
        )



@module.server
def culturomics_view_server(
    input: Inputs, output: Outputs, session: Session, df_culturomics: reactive.Value, df_metagenomics: reactive.Value
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
            metagenomics_dataframe = metagenomics_dataframe[(metagenomics_dataframe['scientific_name'] == selected_organism) & (metagenomics_dataframe['count'] > input.culturomics_count())]

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
    @reactive.event(input.x_axis, input.y_axis, input.experiment_id_2, input.taxrank, input.count, input.color, input.top_selector)
    def plot():
        print(input.top_selector())
        if input.x_axis() != None and input.y_axis() != None:
            dataframe = df.get()
            dataframe = dataframe[(dataframe['experimentid'] == input.experiment_id_2.get()[0]) & (dataframe['taxrank'] == input.taxrank()) & (dataframe['count'] > input.count())]
            # dataframe = dataframe.nlargest(input.top_selector(), columns='count').reset_index(drop=True)
            x_axis_selection = input.x_axis()
            y_axis_selection = input.y_axis()
            fig = (px.histogram(
                data_frame=dataframe,
                x=x_axis_selection,
                y=y_axis_selection,
                color=input.color(),
            )
            .update_layout(
                title="Test Plot",
                yaxis_type='log'
            ).update_xaxes(tickangle=45))
            return fig
    @render_widget
    @reactive.event(input.x_axis, input.y_axis, input.experiment_id_2, input.taxrank, input.count)
    def test_plot():
        if input.x_axis() != None and input.y_axis() != None and len(input.experiment_id_2()) > 0 and df.get().shape[0] != 0:
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
            ui.input_select("column", "Column to Edit", choices=[]),
            ui.input_text("new_value", "New Value"),
            ui.input_action_button("update", "Update", class_="btn-success")
        )

    )

@module.server
def test2_view_server(
    input: Inputs, output: Outputs, session: Session, df: reactive.Value
):
    flag = reactive.value(0)
    @render.text
    def row_count():
        return df.get().shape[0]

    @render.text
    def mean_score():
        pass

    @reactive.effect
    @reactive.event(input.update)
    def updated_df():
        dataframe = df.get()
        selected_column = input.column()
        new_value = input.new_value()
        selected_rows = input.data_selected_rows()
        temp_df = dataframe.copy()
        for row_index in selected_rows:
            dataframe.loc[row_index, selected_column] = new_value
        flag.set(flag.get() + 1)
        df.set(dataframe)
        # df.assign(temp_df)
        # Assuming you are allowing users to edit only one cell at a time:

        # selected_row_index = selected_rows[0]['row_index']
        # df.get().loc[selected_row_index, selected_column] = new_value
        #
        # return df.get()  # Return the modified DataFrame


    @render.data_frame
    def data():
        dataframe = df.get()
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
            ui.card(  # Add a new card
            ui.card_header("Sample selection"),
            ui.output_text_verbatim("first_selected_organism"),
            ui.output_text_verbatim("second_selected_organism"),
            ui.layout_columns(
                ui.input_action_button("send", "Maak dotplot", class_="btn-success", width="200px", height="40px",),
            ),
            ui.layout_columns(
                ui.input_select("jobs", "Select job", choices=['None']),
                ui.input_action_button("goJob", "Go to job", class_="btn-success", width="200px", height="40px", style="margin-top: 20px;"),
            )
        ),
            ui.layout_columns(
                ui.card(
                    ui.card_header("Sample table"),
                    ui.output_data_frame("data"),
                    ui.layout_columns(
                        ui.div(
                            ui.input_action_button("select", "Select", class_="btn-success", width="200px",
                                                   height="40px"),
                            ui.input_action_button("reset", "Reset", class_="btn-danger", width="200px", height="40px"),
                            style="display: flex; justify-content: center; gap: 20px;"
                        ),
                    ),
                ),
                style="margin-top: 20px;",
            ),

        col_widths=(3, 9)

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
        return f"First sample: {second_sample_name.get()}"

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


