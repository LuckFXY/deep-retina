from . import metrics
import numpy as np
functions=_['cc', 'lli', 'rmse', 'fev']
x=np.array([1,2,3],dtype=np.float32)
y=np.array([0.9,1.8,2.6])
for fun in functions:
    getattr(metrics, function)(x,y)