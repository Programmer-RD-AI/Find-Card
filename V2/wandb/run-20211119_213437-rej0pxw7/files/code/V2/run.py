from Model import *

model = Model()
model.train()
for model in models:
    model = Model(model="COCO-Detection/" + model, name="Model-" + model)
    model.train()

# pt = Param_Tunning()
# pt.tune(params)
