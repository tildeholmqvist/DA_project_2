import streamlit as st


class MultiPage:
    """
    A simple multi-page framework for Streamlit apps. Pages are
    registered with add_page() and rendered via a sidebar radio menu
    in run().
    """

    def __init__(self, app_name) -> None:
        """Initialise the app with a name and empty page list."""
        self.pages = []
        self.app_name = app_name
        st.set_page_config(
            page_title=self.app_name,
            page_icon="🚗"
        )

    def add_page(self, title, func) -> None:
        """Register a page with a sidebar title and its render function."""
        self.pages.append({"title": title, "function": func})

    def run(self):
        """Render the app title, sidebar menu, and the selected page."""
        st.title(self.app_name)
        page = st.sidebar.radio(
            "Menu", self.pages, format_func=lambda page: page["title"]
        )
        page["function"]()