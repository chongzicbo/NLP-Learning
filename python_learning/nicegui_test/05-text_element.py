from nicegui import ui

ui.label("some label")
ui.link("NIceGUI", "http://www.baidu.com")
ui.chat_message(
    "Hello NiceGUI!", name="Robot", stamp="now", avatar="https://robohash.org/ui"
)
with ui.element("div").classes("p-2 bg-blue-100"):
    ui.label("inside a colored div")
ui.markdown("This is **Markdown**.")
ui.restructured_text("This is **reStructuredText**.")
ui.mermaid(
    """
graph LR;
    A --> B;
    A --> C;
"""
)
ui.html("This is <strong>HTML</strong>.")

ui.run()
