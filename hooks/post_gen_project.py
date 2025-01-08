import shutil
import os
import sys
import json
from copy import copy
from pathlib import Path
import inspect
import ast
import tokenize
from io import StringIO

# https://github.com/cookiecutter/cookiecutter/issues/824
#   our workaround is to include these utility functions in the CCDS package
from ccds.hook_utils.custom_config import write_custom_config
from ccds.hook_utils.dependencies import basic, packages, scaffold, write_dependencies

#
#  TEMPLATIZED VARIABLES FILLED IN BY COOKIECUTTER
#
packages_to_install = copy(packages)

# {% if cookiecutter.dataset_storage.s3 %}
packages_to_install += ["awscli"]
# {% endif %} #

# {% if cookiecutter.include_code_scaffold == "Yes" %}
packages_to_install += scaffold
# {% endif %}

# {% if cookiecutter.pydata_packages == "basic" %}
packages_to_install += basic
# {% endif %}

# track packages that are not available through conda
pip_only_packages = [
    "awscli",
    "python-dotenv",
]

# Use the selected documentation package specified in the config,
# or none if none selected
docs_path = Path("docs")
# {% if cookiecutter.docs != "none" %}
packages_to_install += ["{{ cookiecutter.docs }}"]
pip_only_packages += ["{{ cookiecutter.docs }}"]
docs_subpath = docs_path / "{{ cookiecutter.docs }}"
for obj in docs_subpath.iterdir():
    shutil.move(str(obj), str(docs_path))
# {% endif %}

# Remove all remaining docs templates
for docs_template in docs_path.iterdir():
    if docs_template.is_dir() and not docs_template.name == "docs":
        shutil.rmtree(docs_template)

#
#  POST-GENERATION FUNCTIONS
#
write_dependencies(
    "{{ cookiecutter.dependency_file }}",
    packages_to_install,
    pip_only_packages,
    repo_name="{{ cookiecutter.repo_name }}",
    module_name="{{ cookiecutter.module_name }}",
    python_version="{{ cookiecutter.python_version_number }}",
)

write_custom_config("{{ cookiecutter.custom_config }}")

# Remove LICENSE if "No license file"
if "{{ cookiecutter.open_source_license }}" == "No license file":
    Path("LICENSE").unlink()

# Make single quotes prettier
# Jinja tojson escapes single-quotes with \u0027 since it's meant for HTML/JS
pyproject_text = Path("pyproject.toml").read_text()
Path("pyproject.toml").write_text(pyproject_text.replace(r"\u0027", "'"))

# {% if cookiecutter.include_code_scaffold == "No" %}
# remove everything except __init__.py so result is an empty package
for generated_path in Path("{{ cookiecutter.module_name }}").iterdir():
    if generated_path.is_dir():
        shutil.rmtree(generated_path)
    elif generated_path.name != "__init__.py":
        generated_path.unlink()
    elif generated_path.name == "__init__.py":
        # remove any content in __init__.py since it won't be available
        generated_path.write_text("")
# {% endif %}


# Extract values from the context
create_notebook_var = "{{ cookiecutter.create_notebook }}"
notebook_name = "{{ cookiecutter.notebook_name }}"
use_yaml_parameters = "{{ cookiecutter.use_yaml_parameters }}"
use_yaml_parameters = use_yaml_parameters.replace("'", '"')
try:
    use_yaml_parameters = json.loads(use_yaml_parameters)
except json.JSONDecodeError as e:
    print(f"Error decoding JSON string: {e}")
file_types_var = "{{ cookiecutter.read_file_types }}"
if list(use_yaml_parameters.keys())[0] == 'Yes':
    yaml_path = use_yaml_parameters['Yes']['yaml_path']
else:
    input_data_path = "{{ cookiecutter.input_data_path }}"
    file_name = "{{ cookiecutter.file_name }}"
use_yaml_parameters = list(use_yaml_parameters.keys())[0] == 'Yes'
file_types_list = [x.strip() for x in file_types_var.split(",")]
print(create_notebook_var)
print(notebook_name)
print(use_yaml_parameters)
print(yaml_path)
print(file_types_list)

def add_yaml_code():
    """
    Hücre kaynağına YAML kodlarını ekler.
    Korner durum eklendi
    """
    current_path = Path(os.getcwd())
    # Load config files from yaml
    cfg = yaml.safe_load(open(current_path / yaml_path, "r"))
    data_path = cfg["paths"]["data"]
    # corner case added
    file_name = cfg["paths"]["file"]
    data_path = Path(current_path) / data_path
    input_data_path = data_path / r"input_data"
    
    return file_name, data_path, cfg, input_data_path

def read_from_CSV(input_data_path, file_name):
    # Read data from CSV                
    csv_file_path = input_data_path / (file_name + ".csv")
    df = pd.read_csv(csv_file_path, header=0, na_values=["---", "Unknown"])
    return csv_file_path, df
    
def add_source_code_to_cell(cell_source, func):
    """Belirtilen fonksiyonun kaynak kodunu hücre kaynağına ekler."""
    
    # Fonksiyonun kaynak kodunu al
    source_code = inspect.getsource(func)
    
    # AST ile docstring ve return ifadelerini kaldır
    tree = ast.parse(source_code)
    function_body = tree.body[0].body
    
    # Docstring'i atla
    if isinstance(function_body[0], ast.Expr) and isinstance(function_body[0].value, ast.Constant):
        function_body = function_body[1:]
    
    # Return ifadelerini atla
    if isinstance(function_body[-1], ast.Return):
        function_body = function_body[:-1]
    
    # AST'den kod bloğunu oluştur
    ast_code = ast.unparse(ast.Module(body=function_body, type_ignores=[]))
    
    # Yorumları korumak için tokenize işlemi
    tokens = tokenize.generate_tokens(StringIO(source_code).readline)
    preserved_lines = []
    for token in tokens:
        token_type, token_string, _, _, _ = token
        if token_type == tokenize.COMMENT: # Yorumları koru
            preserved_lines.append(token_string)
        elif token_type == tokenize.NL: # Boş satırları ekle
            preserved_lines.append("")
    
    # Yorumları ve AST kodunu birleştir
    combined_code = "\n".join(preserved_lines) + "\n" + ast_code
    
    # Fazla boş satırları kaldır
    cleaned_code = "\n".join(line for line in combined_code.splitlines() if line.strip())
    
    # Hücre kaynağına ekle
    cell_source.append(cleaned_code)
    return cell_source
    
if create_notebook_var == 'Yes' or True:
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# {{ cookiecutter.notebook_name }} Modeling" 
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Dependencies"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "from sklearn.calibration import LabelEncoder\n",
                    "import yaml\n",
                    "import os\n",
                    "from pathlib import Path\n",
                    "import numpy as np\n",
                    "\n",
                    f"input_data_path = {input_data_path}\n" if use_yaml_parameters == 'No' else "",
                    f"file_name = {file_name}\n" if use_yaml_parameters == 'No' else "",
                    f"yaml_path = {yaml_path}\n" if use_yaml_parameters == 'Yes' else "",
                    "\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "\n",
                    "# Function to clean column names\n",
                    "def clean_column_name(column_name):\n",
                    "    return column_name.strip().replace('[', '').replace(']', '').replace(' ', '_').replace('?', '').replace('/','per')\n",
                    "\n",
                    "# Clean column names\n",
                    "df.columns = [clean_column_name(col) for col in df.columns]\n",
                    "\n",
                    "columns_to_drop = []\n",
                    "if cfg[\"feature_selection\"][\"columns_to_drop\"] is not None:\n",
                    "    columns_to_drop = cfg[\"feature_selection\"][\"columns_to_drop\"]\n",
                    "\n",
                    "df = df.drop(columns=columns_to_drop)"
                ]
            },
            {
             "cell_type": "markdown",
             "metadata": {},
             "source": [
              "### 1.2. Data Overview - Numerical"
             ]
            },
            {
             "cell_type": "code",
             "execution_count": None,
             "metadata": {},
             "outputs": [],
             "source": [
              "print(f\"\\nDATASET SUMMARY\\n\")\n",
              "print(df.describe(include=\"all\"), \"\\n\")\n",
              "print(df.info(), \"\\n\")"
             ]
            },
            {
             "cell_type": "markdown",
             "metadata": {},
             "source": [
              "***Review***\n",
              "\n",
              "By checking the min-max values of numeric columns, the number of missing values, and the unique values of categorical features, we can identify whether there are logically inconsistent or questionable data points.\n",
              "Here are the list of features which don't have enough data to represent (# of missing points > 0.5 * number of total rows):\n",
              "\n",
              "* Measurment_ID\n",
              "* TestBedperTestBench\n",
              "* pipeDiameter_mm\n",
              "* VehicleMarket_-\n",
              "* VehicleMax._laden_mass_kg\n",
              "* VehicleMax._vehicle_load_(MVL)_kg\n",
              "* DiscType_-\n",
              "* DiscSurface_-\n",
              "* Piston_Area_cm^2\n",
              "\n",
              "These will be removed from the dataframe due to high number of missing values."
             ]
            }
        ],
        "metadata": {
            # ... metadata bilgileri ...
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    if use_yaml_parameters == 'Yes':
        notebook_content["cells"][2]["source"] = add_source_code_to_cell(notebook_content["cells"][2]["source"], add_yaml_code)

    if "csv" in file_types_list:
        notebook_content["cells"][2]["source"] = add_source_code_to_cell(notebook_content["cells"][2]["source"], read_from_CSV)
        
    with open('C:\\Users\\u27f79\\.cookiecutters\\cookiecutter-data-science\\deneme.ipynb', 'w') as f:
        json.dump(notebook_content, f)
