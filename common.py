import os

project_path = os.path.abspath(os.path.dirname(__file__))
project_path = project_path[:project_path.find('pyrdw') + len('pyrdw')]
root_path = os.path.dirname(project_path)
data_path = root_path + '\\pyrdwdata'