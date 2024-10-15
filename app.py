from st_pages import Page, show_pages, add_page_title
add_page_title()

show_pages(
    [
        Page("app.py", "Home", "🏠"),
        Page("main.py", "Page 2", ":books:"),
        Page("chat.py", "Page 2", ":books:"),

    ]
)