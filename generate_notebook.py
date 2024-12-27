import json
import os

def create_notebook(create_notebook, notebook_name, add_title):
    print("create_notebook;" + str(create_notebook))
    if create_notebook == 'Yes' or True:
        notebook_content = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "# Brake Emission Modeling" if add_title == 'Yes' else ""
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
                        "\n",
                        "current_path = Path(os.getcwd())\n",
                        "# Load config files from yaml\n",
                        "cfg = yaml.safe_load(open(current_path / \"config/config.yaml\", \"r\"))\n",
                        "data_path = cfg[\"paths\"][\"data\"]\n",
                        "file_name = cfg[\"paths\"][\"file\"]\n",
                        "data_path = Path(current_path) / data_path\n",
                        "\n",
                        "# Read data from CSV\n",
                        "input_data_path = data_path / r\"input_data\"\n",
                        "csv_file_path = input_data_path / (file_name + \".csv\")\n",
                        "df = pd.read_csv(csv_file_path, header=0, na_values=[\"---\", \"Unknown\"])\n",
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
        with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'tests', 'deneme.ipynb'), 'w') as f:
            json.dump(notebook_content, f)

if __name__ == "__main__":
     # Path to the cookiecutter context JSON file
    context_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), "ccds.json")
    
    # Read the cookiecutter context
    with open(context_file, 'r') as f:
        context = json.load(f)
    
    # Extract values from the context
    create_notebook_var = context.get("create_notebook", "No")
    notebook_name = context.get("notebook_name", "notebook")
    add_title = context.get("add_title", "No")
    
    # Call the function to create the notebook
    create_notebook(create_notebook_var, notebook_name, add_title)
