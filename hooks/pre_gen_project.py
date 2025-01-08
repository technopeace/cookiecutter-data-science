# Functions here run before the project is generated.
import sys

def main():
    # Kullanıcının `use_yaml_parameters` seçimini al
    use_yaml_parameters = "{{ cookiecutter.use_yaml_parameters }}"
    
    # Eğer 'No' seçilmişse, kullanıcıya YAML yolu sormayı atla
    if use_yaml_parameters.lower() == "no":
        sys.stdout.write("Skipping YAML path as 'use_yaml_parameters' is 'No'.\n")
        # Burada, `yaml_path` boş bir değerle geçersiz kılınabilir.
        cookiecutter_context = {
            "yaml_path": ""
        }

if __name__ == "__main__":
    main()

# For the use of these hooks, see
# See https://cookiecutter.readthedocs.io/en/1.7.2/advanced/hooks.html
