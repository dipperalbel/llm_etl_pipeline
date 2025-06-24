import os
import sys

project = "ETL Project"
copyright = "Alberto Bellumat"
author = "Alberto Bellumat"
release = "0.1.0"


# Add path to the package
sys.path.insert(0, os.path.abspath("../.."))


# Skip Pydantic internal methods and attributes from the API docs
def skip_pydantic_internals(app, what, name, obj, skip, options):
    if name.startswith("model_") or name in [
        "schema",
        "schema_json",
        "construct",
        "copy",
        "dict",
        "json",
        "validate",
        "update_forward_refs",
        "parse_obj",
        "parse_file",
        "parse_raw",
        "from_orm",
        "Config",
        "__fields__",
        "__annotations__",
        "__config__"
    ]:
        return True
    return skip


def setup(app):
    app.connect("autodoc-skip-member", skip_pydantic_internals)


# Extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx.ext.autosectionlabel",
    "sphinx_design",
    "sphinxext.opengraph",
    "sphinx_sitemap",
]

autosectionlabel_prefix_document = True

# Autodoc settings
autodoc_member_order = "bysource"
autoclass_content = "both"
autodoc_typehints = "description" 
typehints_fully_qualified = True
typehints_document_rtype = False

templates_path = ["_templates"]
exclude_patterns = []

html_title = f"{project} Documentation"
html_theme = "sphinx_rtd_theme"
master_doc = "index"
html_baseurl = "https://dipperalbel.github.io/llm_etl_pipeline/"