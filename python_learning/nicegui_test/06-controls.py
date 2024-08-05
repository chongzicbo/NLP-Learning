from nicegui import ui

ui.button("Click me!", on_click=lambda: ui.notify("You clicked me!"))
with ui.button_group():
    ui.button("One", on_click=lambda: ui.notify("You clicked Button 1!"))
    ui.button("Two", on_click=lambda: ui.notify("You clicked Button 2!"))
    ui.button("Three", on_click=lambda: ui.notify("You clicked Button 3!"))

with ui.dropdown_button("Open me!", auto_close=True):
    ui.item("Item 1", on_click=lambda: ui.notify("You clicked item 1"))
    ui.item("Item 2", on_click=lambda: ui.notify("You clicked item 2"))
with ui.button("Click me!", on_click=lambda: badge.set_text(int(badge.text) + 1)):
    badge = ui.badge("0", color="red").props("floating")
with ui.row().classes("gap-1"):
    ui.chip("Click me", icon="ads_click", on_click=lambda: ui.notify("Clicked"))
    ui.chip("Selectable", selectable=True, icon="bookmark", color="orange")
    ui.chip("Removable", removable=True, icon="label", color="indigo-3")
    ui.chip("Styled", icon="star", color="green").props("outline square")
    ui.chip("Disabled", icon="block", color="red").set_enabled(False)
toggle1 = ui.toggle([1, 2, 3], value=1)
toggle2 = ui.toggle({1: "A", 2: "B", 3: "C"}).bind_value(toggle1, "value")
radio1 = ui.radio([1, 2, 3], value=1).props("inline")
radio2 = ui.radio({1: "A", 2: "B", 3: "C"}).props("inline").bind_value(radio1, "value")
ui.run()
