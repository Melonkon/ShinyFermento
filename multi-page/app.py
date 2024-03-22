from pathlib import Path

import pandas as pd
import psycopg2
from modules import data_view_server, data_view_ui, training_server, training_ui

from shiny import App, Inputs, Outputs, Session, reactive, ui

df = pd.read_csv(Path(__file__).parent / "scores.csv")


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

app_ui = ui.page_navbar(
    training_ui("tab1"),
    data_view_ui("tab2"),
    sidebar=ui.sidebar(
        ui.input_select(
            "account",
            "Account",
            choices=[
                "Berge & Berge",
                "Fritsch & Fritsch",
                "Hintz & Hintz",
                "Mosciski and Sons",
                "Wolff Ltd",
            ],
        ),
        width="300px",
    ),
    header=ui.include_css(Path(__file__).parent / "styles.css"),
    id="tabs",
    title="Monitoring",
)


def server(input: Inputs, output: Outputs, session: Session):

    @reactive.calc()
    def filtered_data() -> pd.DataFrame:
        return df.loc[df["account"] == input.account()]

    training_server(id="tab1", df=filtered_data)
    data_view_server(id="tab2", df=filtered_data)


app = App(app_ui, server)
