from nicegui import ui

columns = [
    {
        "name": "name",
        "label": "Name",
        "field": "name",
        "required": True,
        "align": "left",
    },
    {"name": "age", "label": "Age", "field": "age", "sortable": True},
]
rows = [
    {"name": "Alice", "age": 18},
    {"name": "Bob", "age": 21},
    {"name": "Carol"},
]
ui.table(columns=columns, rows=rows, row_key="name")
grid = ui.aggrid(
    {
        "defaultColDef": {"flex": 1},
        "columnDefs": [
            {"headerName": "Name", "field": "name"},
            {"headerName": "Age", "field": "age"},
            {"headerName": "Parent", "field": "parent", "hide": True},
        ],
        "rowData": [
            {"name": "Alice", "age": 18, "parent": "David"},
            {"name": "Bob", "age": 21, "parent": "Eve"},
            {"name": "Carol", "age": 42, "parent": "Frank"},
        ],
        "rowSelection": "multiple",
    }
).classes("max-h-40")


def update():
    grid.options["rowData"][0]["age"] += 1
    grid.update()


ui.button("Update", on_click=update)
ui.button("Select all", on_click=lambda: grid.run_grid_method("selectAll"))
ui.button(
    "Show parent",
    on_click=lambda: grid.run_column_method("setColumnVisible", "parent", True),
)
ui.run()
