from pathlib import Path

import pandas as pd
import psycopg2
from modules import test_view_ui, test_view_server, test2_view_ui, test2_view_server, dotplot_view_ui, dotplot_view_server

from shiny import App, Inputs, Outputs, Session, reactive, ui

class DatabaseManager:
    def __init__(self, db_config):
        self.connection = psycopg2.connect(**db_config)
        self.cursor = self.connection.cursor()

    def execute(self, command, params=None):
        if params:
            self.cursor.execute(command, params)
        else:
            self.cursor.execute(command)

    def fetch_all(self):
        return self.cursor.fetchall()

    def rollback(self):
        self.connection.rollback()

    def commit(self):
        self.connection.commit()

    def close(self):
        self.cursor.close()
        self.connection.close()

# db_config = {
#     "dbname": "test2",
#     "user": "postgres",
#     "password": "4321or1234",
#     "host": "localhost",
#     "port": "5433"
# }

db_config = {
    "dbname": "test2",
    "user": "ridderplaat.q",
    "password": "Quinten4321or1234!",
    "host": "145.97.18.149",
    "port": "5432"
}

app_ui = ui.page_navbar(
    dotplot_view_ui("tab1"),
    test_view_ui("tab2"),
    test2_view_ui("tab3"),

    sidebar=ui.sidebar(
        ui.input_select(
            "tables",
            "Tables",
            choices=[
                "bron"
            ],
        ),
        ui.input_selectize(
                    "columns",
                    "Columns",
                    choices=[
                        "bron"
                    ],
            multiple=True,
                ),
        width="300px",
    ),
    header=ui.include_css(Path(__file__).parent / "styles.css"),
    id="tabs",
    title="Monitoring",
)


def server(input: Inputs, output: Outputs, session: Session):

    db_manager = DatabaseManager(db_config)

    dataframe = reactive.value(pd.DataFrame())
    dataframe_meegeven = reactive.value(pd.DataFrame())
    dataframe_culturomics = reactive.value(pd.DataFrame())
    @reactive.effect
    def all_tables():
        db_manager.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
        result = db_manager.fetch_all()
        tables = [table_name[0] for table_name in result]
        ui.update_select("tables", choices=tables)

    @reactive.effect
    def df_culturomics_aanmaken():
        selected_table = "corganism"
        if dataframe.get().shape[0] != 0:
            try:
                db_manager.execute(f"SELECT * FROM {selected_table}")
                result = db_manager.fetch_all()
                columns = db_manager.cursor.description
                dataframe_culturomics.set(pd.DataFrame(result, columns=[column[0] for column in columns]))
                return pd.DataFrame(result, columns=[column[0] for column in columns])
            except Exception as e:
                # Handle database errors here (display an error message, etc.)
                print(f"Error loading data: {e}")

    @reactive.effect
    def columns():
        columns = dataframe.get().columns.tolist()
        ui.update_selectize("columns", choices=columns, selected=columns)

    @reactive.effect
    @reactive.event(input.columns)
    def df_aanpassen():
        if dataframe.get().shape[0] != 0 and len(input.columns()) != 0:
            # dataframe_meegeven = dataframe.get()
            dataframe_meegeven.set(dataframe.get()[list(input.columns())])
            # dataframe.set(dataframe.get()[input.columns()])
            # print(dataframe.get().columns)



    @reactive.effect
    def df_aanmaken():  # Load directly from the database
        selected_table = input.tables()
        try:
            db_manager.execute(f"SELECT * FROM {selected_table}")
            result = db_manager.fetch_all()
            columns = db_manager.cursor.description
            dataframe.set(pd.DataFrame(result, columns=[column[0] for column in columns]))
            return pd.DataFrame(result, columns=[column[0] for column in columns])
        except Exception as e:
            # Handle database errors here (display an error message, etc.)
            print(f"Error loading data: {e}")

    dotplot_view_server(id="tab1", df=dataframe_culturomics)
    test_view_server(id="tab2", df=dataframe_meegeven)
    test2_view_server(id="tab3", df=dataframe_meegeven)

    # training_server(id="tab2", df=filtered_data)
    # data_view_server(id="tab3", df=filtered_data)





app = App(app_ui, server)
