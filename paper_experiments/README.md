# Hyperparameter settings

The same settings were used for all datasets. Hyperparameter optimization could further improve
the performance and will be in the scope of future research.

### P3LS
- No. latent variables: 10

### PPSR
- Population size: 1000
- Number of generations: 200
- Fitness function: MSE (regression tasks)/ Accuracy (classification tasks)
- Initialization method: Half and half
- Non-terminals: +, - , *
- Terminals: variables, constant 
- Mutation method: Uniform mutation
- Mutation probability: 0.2 
- Crossover method: One-point crossover
- Crossover probability: 0.95
- Minimum subtree depth: 0 
- Maximum subtree depth: 2
- Maximum depth: 15
- Maximum length: 120
- Selection type: Tournament
- Tournament size: 5

### Secureboost
- Learning rate: 0.1
- Number of boosting rounds: 200
- Row subsample by tree: 0.8
- Column subsample by tree: 0.8
- Objective function: linear (regression tasks)/ logistic (classification tasks)
- Maximum depth: 3

### SplitNN

Base model

````angular2html
base_model = keras.Sequential(
    [keras.Input(shape=input_dim),
     layers.Dense(32, activation="relu"),
     layers.Dense(output_dim, activation="relu")
    ]
)


base_model.compile(
    loss="mean_squared_error",
    optimizer="adam",
    metrics=["mse"]
)
````


Fuse model

````angular2html
input_layers = []
for i in range(party_nums):
    input_layers.append(
        keras.Input(
            input_dim,
        )
    )

merged_layer = layers.concatenate(input_layers)
fuse_layer_1 = layers.Dense(32, activation="relu")(merged_layer)
fuse_layer_2 = layers.Dense(16, activation="relu")(fuse_layer_1)
fuse_layer_3 = layers.Dense(8, activation="relu")(fuse_layer_2)
output = layers.Dense(output_dim)(fuse_layer_3)

fuse_model = keras.Model(inputs=input_layers, outputs=output)

# Compile model
fuse_model.compile(
    loss="mean_squared_error",
    optimizer="adam",
    metrics=["mse"],
)
````