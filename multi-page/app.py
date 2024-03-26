from pathlib import Path

import pandas as pd
import psycopg2
from modules import test_view_ui, test_view_server

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

db_config = {
    "dbname": "test",
    "user": "postgres",
    "password": "4321or1234",
    "host": "localhost",
    "port": "5432"
}

app_ui = ui.page_navbar(
    test_view_ui("tab1"),
    sidebar=ui.sidebar(
        ui.input_select(
            "tables",
            "Tables",
            choices=[
                "bron"
            ],
        ),
        width="300px",
    ),
    header=ui.include_css(Path(__file__).parent / "styles.css"),
    id="tabs",
    title="Monitoring",
)


def server(input: Inputs, output: Outputs, session: Session):
    db_manager = DatabaseManager(db_config)

    @reactive.effect
    def all_tables():
        db_manager.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
        result = db_manager.fetch_all()
        tables = [table_name[0] for table_name in result]
        ui.update_select("tables", choices=tables)

    @reactive.calc
    def df():  # Load directly from the database
        selected_table = input.tables()
        try:
            db_manager.execute(f"SELECT * FROM {selected_table}")
            result = db_manager.fetch_all()
            columns = db_manager.cursor.description
            return pd.DataFrame(result, columns=[column[0] for column in columns])
        except Exception as e:
            # Handle database errors here (display an error message, etc.)
            print(f"Error loading data: {e}")
            return pd.DataFrame()  # Return an empty DataFrame on error





    test_view_server(id="tab1", df=df)
    # training_server(id="tab2", df=filtered_data)
    # data_view_server(id="tab3", df=filtered_data)




app = App(app_ui, server)
