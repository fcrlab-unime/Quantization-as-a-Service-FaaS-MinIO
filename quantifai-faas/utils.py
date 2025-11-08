import importlib.util
import os
import shlex
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

def load_class_from_file(class_name, module_path, file_path, **kwargs):
    """
    Dynamically load a class from a Python file.
    Args:
        class_name (str): Name of the class to load.
        module_path (str): Directory containing the file.
        file_path (str): Full path to the Python file.
        kwargs: Additional arguments for the class constructor.
    Returns:
        object: An instance of the loaded class.
    """
    try:
        logging.debug(f"Loading class '{class_name}' from file: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        spec = importlib.util.spec_from_file_location("module.name", file_path)
        if spec is None:
            raise ImportError(f"Unable to load spec for file: {file_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        logging.debug(f"Loaded module from file: {file_path}")

        # Safely get the class
        if not hasattr(module, class_name):
            raise AttributeError(f"Class '{class_name}' not found in module {file_path}")

        class_ = getattr(module, class_name)
        if not callable(class_):
            raise TypeError(f"'{class_name}' is not a valid class.")

        # Log arguments being passed to the class
        logging.debug(f"Arguments passed to '{class_name}': {kwargs}")
        return class_(**kwargs)
    except Exception as e:
        logging.error("Error in load_class_from_file.", exc_info=True)
        raise


def parse_dynamic_args(args):
    """
    Parses dynamic arguments from a string.
    Args:
        args (str): Arguments in the form 'key1=value1 key2=value2'.
    Returns:
        dict: Parsed arguments as key-value pairs.
    """
    try:
        logging.debug(f"Parsing dynamic args: {args}")
        # Use shlex to split args while respecting quotes
        args_list = shlex.split(args)
        parsed_args = {}
        for arg in args_list:
            if "=" in arg:
                key, value = arg.split("=", 1)
                parsed_args[key.strip()] = value.strip()
            else:
                raise ValueError(f"Invalid argument format: '{arg}'. Expected 'key=value'.")
        # Convert numerical values where applicable
        for key, value in parsed_args.items():
            if value.isdigit():
                parsed_args[key] = int(value)
            else:
                try:
                    parsed_args[key] = float(value)
                except ValueError:
                    pass
        logging.debug(f"Parsed args: {parsed_args}")
        return parsed_args
    except Exception as e:
        logging.error("Error parsing dynamic args.", exc_info=True)
        raise
