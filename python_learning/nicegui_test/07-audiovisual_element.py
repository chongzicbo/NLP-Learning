from nicegui import ui

ui.image("https://picsum.photos/id/377/640/360")
with ui.image("https://picsum.photos/id/29/640/360"):
    ui.label("Nice!").classes("absolute-bottom text-subtitle2 text-center")

with ui.image("https://cdn.stocksnap.io/img-thumbs/960w/airplane-sky_DYPWDEEILG.jpg"):
    ui.html(
        """
        <svg viewBox="0 0 960 638" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">
        <circle cx="445" cy="300" r="100" fill="none" stroke="red" stroke-width="10" />
        </svg>
    """
    ).classes("w-full bg-transparent")
ui.run()
