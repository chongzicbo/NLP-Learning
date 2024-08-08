from nicegui import ui


class Demo:
    def __init__(self):
        self.number = 1


demo = Demo()
v = ui.checkbox("visible", value=True)
with ui.column().bind_visibility_from(v, "value"):
    ui.slider(min=1, max=3).bind_value(demo, "number")
    ui.toggle({1: "A", 2: "B", 3: "C"}).bind_value(demo, "number")
    ui.number().bind_value(demo, "number")

data = {"name": "Bob", "age": 17}

ui.label().bind_text_from(data, "name", backward=lambda n: f"Name: {n}")
ui.label().bind_text_from(data, "age", backward=lambda a: f"Age: {a}")

ui.button("Turn 18", on_click=lambda: data.update(age=18))
ui.run()
