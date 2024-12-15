from contextlib import contextmanager

from frontend.components.menu import menu

from nicegui import ui


@contextmanager
def frame():
    """Custom page frame to share the same styling and behavior across all pages"""
    ui.colors(
        primary="#6E93D6", secondary="#53B689", accent="#111B1E", positive="#53B689"
    )
    with ui.header():
        ui.image("static/logo_bayes.webp").classes("h-8 w-8 p-0 m-0")
        # ui.label('Modularization Example').classes('font-bold')
        with ui.row().classes("my-auto text-xl"):
            menu()
    with ui.row().classes("h-full m-0 p-0"):
        with ui.column().classes("w-64 border-r h-full p-4"):
            ui.button(color="secondary").classes("w-full h-12 m-2 p-2")
        with ui.column().classes("absolute-center items-center"):
            yield
