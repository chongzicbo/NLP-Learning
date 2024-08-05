from nicegui import ui
from uuid import uuid4


@ui.page("/other_page")
def other_page():
    ui.label("Welcome to the other side")


@ui.page("/dark_page", dark=True)
def dark_page():
    ui.label("Welcome to the dark side")


ui.link("Visit other page", other_page)
ui.link("Visit dark page", dark_page)


@ui.page("/private_page")
async def private_page():
    ui.label(f"private page with ID {uuid4()}")


ui.label(f"shared auto-index page with ID {uuid4()}")
ui.link("private page", private_page)


@ui.page("/page_layout")
def page_layout():
    ui.label("CONTENT")
    [ui.label(f"Line {i}") for i in range(100)]
    with ui.header(elevated=True).style("background-color: #3874c8").classes(
        "items-center justify-between"
    ):
        ui.label("HEADER")
        ui.button(on_click=lambda: right_drawer.toggle(), icon="menu").props(
            "flat color=white"
        )
    with ui.left_drawer(top_corner=True, bottom_corner=True).style(
        "background-color: #d7e3f4"
    ):
        ui.label("LEFT DRAWER")
    with ui.right_drawer(fixed=False).style("background-color: #ebf1fa").props(
        "bordered"
    ) as right_drawer:
        ui.label("RIGHT DRAWER")
    with ui.footer().style("background-color: #3874c8"):
        ui.label("FOOTER")


ui.link("show page with fancy layout", page_layout)


import random
from nicegui import app, ui


@app.get("/random/{max}")
def generate_random_number(max: int):
    return {"min": 0, "max": max, "value": random.randint(0, max)}


max = ui.number("max", value=100)
ui.button(
    "generate random number",
    on_click=lambda: ui.navigate.to(f"/random/{max.value:.0f}"),
)
ui.run()
