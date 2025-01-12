from sklearn.model_selection import ParameterGrid

param_grid = {
    "param1": ["value1", "value2", "value3"],
    "paramN": ["value1", "value2", "valueM"],
}

grid = ParameterGrid(param_grid)

for params in grid:
    print(params)
