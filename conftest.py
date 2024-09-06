import pytest
import sys
import os

# Añade el directorio raíz del proyecto al PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture(scope="session")
def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

@pytest.fixture(scope="session")
def data_dir(project_root):
    return os.path.join(project_root, 'data')

@pytest.fixture(scope="session")
def notebooks_dir(project_root):
    return os.path.join(project_root, 'notebooks')