import frontend.themes.main_theme as theme
from frontend.components.message import message

from nicegui import ui


def create() -> None:
    @ui.page("/a")
    def page_a():
        with theme.frame():
            message("Page A")
            ui.label("This page is defined in a function.")
