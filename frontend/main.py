#!/usr/bin/env python3
from frontend.pages import home_page, page_a
import frontend.themes.main_theme as theme

from nicegui import app, ui


# Example 1: use a custom page decorator directly and putting the content creation into a separate function
@ui.page('/')
def index_page() -> None:
    with theme.frame():
        
        home_page.content()



page_a.create()


ui.run(title='Modularization Example')