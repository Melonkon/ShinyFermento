from typing import Callable

import pandas as pd
from plots import plot_auc_curve, plot_precision_recall_curve, plot_score_distribution

from shiny import Inputs, Outputs, Session, module, render, ui, reactive


@module.ui
def test_view_ui():
    return ui.nav_panel(
        "View Data",
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
def test_view_server(
    input: Inputs, output: Outputs, session: Session, df: pd.DataFrame
):
    flag = reactive.value(0)
    @render.text
    def row_count():
        return df().shape[0]

    @render.text
    def mean_score():
        pass

    @reactive.effect
    @reactive.event(input.update)
    def updated_df():
        selected_column = input.column()
        new_value = input.new_value()
        selected_rows = input.data_selected_rows()
        temp_df = df().copy()

        print(selected_column, new_value, selected_rows)
        print(df().loc[selected_rows, selected_column])
        df().loc[selected_rows, selected_column] = new_value
        flag.set(flag.get() + 1)
        # df.assign(temp_df)
        # Assuming you are allowing users to edit only one cell at a time:

        # selected_row_index = selected_rows[0]['row_index']
        # df().loc[selected_row_index, selected_column] = new_value
        #
        # return df()  # Return the modified DataFrame


    @render.data_frame
    def data():
        flag.get()
        print(flag.get())
        return render.DataGrid(df(), row_selection_mode="multiple", filters=True, summary=True)

    @render.ui()
    def rows():
        rows = input.data_selected_rows()
        selected = ", ".join(str(i) for i in sorted(rows)) if rows else "None"
        return f"{selected}"

    @reactive.effect
    def column():
        choices = df().columns.tolist()
        ui.update_select("column", choices=choices)
