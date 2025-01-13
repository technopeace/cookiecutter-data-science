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
from collections import OrderedDict
import ipywidgets as widgets
import time
import threading

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

def extractDataFromCookieCutter(parameter, yes_parameter_name="", no_parameter_name=""):
    parameter = parameter.replace("'", '"')
    try:
        parameter = json.loads(parameter)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON string: {e}")
        return None, None, None
    yes_parameter_var = parameter.get('Yes', {}).get(yes_parameter_name, None)
    no_parameter_var = parameter.get('No', {}).get(no_parameter_name, None)
    parameter = list(parameter.keys())[0]
    if yes_parameter_var is None:
        print(f"Warning: 'Yes' key or '{yes_parameter_name}' not found in parameter.")
    if no_parameter_var is None:
        print(f"Warning: 'No' key or '{no_parameter_name}' not found in parameter.")
    return parameter, yes_parameter_var, no_parameter_var

# Dosyayı okuyan ve Output widget'ını güncelleyen fonksiyon
def dosyayi_oku_ve_goster(dosya_adi):
    out = widgets.Output()
    with out:
        while True:
          try:
            with open(dosya_adi, 'r', encoding='utf-8') as f:
              icerik = f.read()
              widgets.Output() = icerik
              out.clear_output(wait=True)  # Önceki çıktıyı temizle
              print(icerik)
          except FileNotFoundError:
            out.clear_output(wait=True)
            print(f"{dosya_adi} dosyası bulunamadı.")
          time.sleep(1)  # 1 saniye bekle


def process_nested_keys(parameter, base_key, prefix="", output_dict = None):
    """
    JSON içindeki her bir anahtar ve alt anahtarın değerlerini derinlemesine işleyerek yazdırır.

    Args:
        parameter (dict or list or str): JSON verisi
        prefix (str): Anahtarın başına eklenecek yol bilgisi

    Returns:
        None
    """
    if output_dict is None:
        output_dict = {}
    if isinstance(parameter, dict):
        for key, value in parameter.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            process_nested_keys(value, base_key, new_prefix, output_dict)
    elif isinstance(parameter, list):
        for idx, item in enumerate(parameter):
            new_prefix = f"{prefix}[{idx}]"
            process_nested_keys(item, base_key, new_prefix, output_dict)
    else:
        print(f"{prefix}: {parameter}")
        output_dict[base_key] = prefix.split(".")[0]
        output_dict[prefix.split(".")[1]] = parameter
    return output_dict



use_yaml_parameters  = process_nested_keys(ast.literal_eval("{{ cookiecutter.use_yaml_parameters }}"), "use_yaml_parameters", output_dict={})
remove_strange_chars_from_column  = process_nested_keys(ast.literal_eval("{{ cookiecutter.remove_strange_chars_from_column }}"), "remove_strange_chars_from_column", output_dict={})
    
# Extract values from the context
create_notebook_var = "{{ cookiecutter.create_notebook }}"
notebook_name = "{{ cookiecutter.notebook_name }}"
#use_yaml_parameters, yaml_path, input_data_path = extractDataFromCookieCutter("{{ cookiecutter.use_yaml_parameters }}", "yaml_path", "input_data_path")
#use_yaml_parameters, yaml_path, file_name = extractDataFromCookieCutter("{{ cookiecutter.use_yaml_parameters }}", "yaml_path", "file_name")

file_types_var = "{{ cookiecutter.read_file_types }}"

#remove_strange_chars_from_column, strange_chars_var, _ = extractDataFromCookieCutter("{{ cookiecutter.remove_strange_chars_from_column }}", "strange_chars")

file_types_list = [x.strip() for x in file_types_var.split(",")]
replace_chars = [x.strip() for x in remove_strange_chars_from_column.get("remove_strange_chars_from_column", "").split(",")]
replace_chars = [char if char != '' else ' ' for char in replace_chars]


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

def clean_column_of_df(df, replace_chars):
    # Function to clean column names
    def clean_column_name(column_name, replace_chars):
        for char in replace_chars:
            if char == '/':
                column_name = column_name.replace(char, 'per')
            elif char == " ":
                column_name = column_name.replace(char, '_')
            else:
                column_name = column_name.replace(char, '')
        return column_name

    # Clean column names
    df.columns = [clean_column_name(col, replace_chars) for col in df.columns]
    return df

def take_columns_from_yaml_to_drop_func(df, cfg):
    columns_to_drop = []
    if cfg["feature_selection"]["columns_to_drop"] is not None:
        columns_to_drop = cfg["feature_selection"]["columns_to_drop"]
    
    df = df.drop(columns=columns_to_drop)
    return df

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
    
if create_notebook_var == 'Yes':
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
                    f"input_data_path = '{use_yaml_parameters.get('input_data_path', '')}'\n" if use_yaml_parameters.get("use_yaml_parameters", "") == 'No' else "",
                    f"file_name = '{use_yaml_parameters.get('file_name', '')}'\n" if use_yaml_parameters.get("use_yaml_parameters", "") == 'No' else "",
                    f"yaml_path = '{use_yaml_parameters.get('yaml_path', '')}'\n" if use_yaml_parameters.get("use_yaml_parameters", "") == 'Yes' else "",
                    f"replace_chars = '{remove_strange_chars_from_column.get('strange_chars', '')}'\n" if remove_strange_chars_from_column.get("remove_strange_chars_from_column", "") == 'Yes' else "",
                    "\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "\n"
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
              
             ]
            }
        ],
        "metadata": {
            # ... metadata bilgileri ...
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    if use_yaml_parameters.get("use_yaml_parameters", "") == 'Yes':
        notebook_content["cells"][2]["source"] = add_source_code_to_cell(notebook_content["cells"][2]["source"], add_yaml_code)

    if "csv" in file_types_list:
        notebook_content["cells"][2]["source"] = add_source_code_to_cell(notebook_content["cells"][2]["source"], read_from_CSV)

    if remove_strange_chars_from_column.get("remove_strange_chars_from_column", "") == 'Yes':
        notebook_content["cells"][2]["source"] = add_source_code_to_cell(notebook_content["cells"][2]["source"], clean_column_of_df)

    if use_yaml_parameters.get("use_yaml_parameters", "") == 'Yes':
        notebook_content["cells"][3]["source"] = add_source_code_to_cell(notebook_content["cells"][3]["source"], take_columns_from_yaml_to_drop_func)

    # .txt dosyanızın adını buraya yazın
    txt_dosyasi = 'C:\\Users\\u27f79\\.cookiecutters\\cookiecutter-data-science\\aciklama.txt' 

    
    # Fonksiyonu ayrı bir thread'de çalıştırın
    thread = threading.Thread(target=dosyayi_oku_ve_goster, args=(txt_dosyasi,))
    thread.start()

    # Markdown hücresine kod hücresinin çıktısını ekleyin
    notebook_content["cells"][4]["source"] = [
        "%%capture output\n",
        "import ipywidgets as widgets\n",
        "widgets.VBox([out])\n",
        "output.show()\n"
    ]
    
    
    with open('C:\\Users\\u27f79\\.cookiecutters\\cookiecutter-data-science\\deneme.ipynb', 'w') as f:
        json.dump(notebook_content, f)
