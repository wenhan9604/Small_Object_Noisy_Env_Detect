## Output Models
### RCAN
#### 1.
```
train:  
  batch_size: 16  
  lr: 0.0001  
  n_epochs: 10  

network:
  model: 'rcan'

optimizer:
  type: "sgd"
  weight_decay: 0.0
```

#### 2.
Epoch [40/40], Loss: 0.0608  
Epoch [40] Validation Loss: 0.0601
```
train:
  batch_size: 16
  lr: 0.0001
  n_epochs: 40 

network:
  model: 'rcan'

optimizer:
  type: "sgd"
  weight_decay: 0.0
```