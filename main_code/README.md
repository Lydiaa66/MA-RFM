# Main Code Layout

Common Python modules are placed directly under `main_code/2d` and `main_code/3d`.
Example-specific modules remain in subdirectories such as `main_code/2d/Ex4.4` or `main_code/3d/Ex4.7`.

Notebook import order is now:

1. example-specific path, for example `../main_code/2d/Ex4.4`
2. common path, for example `../main_code/2d`

This keeps special-case code local while letting shared modules live in one place.
