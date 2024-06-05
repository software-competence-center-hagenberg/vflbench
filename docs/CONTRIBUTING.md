### Prerequisites

- The dataset must have natural borders that can be used to naturally split the features
into different groups.
- The dataset must have been preprocessed.

### Steps

- Load data
- Assign features to different parties
- Perform train/test split
- Save data into a pickle file using the following format

    ````
    fed_data = {
        "num_parties": num_parties,

        "task": "regression", # "classification"

        "cross_valid_data": [Xs_train, y_train, Xs_test, y_test]
    }
    ````
  
    ````
    Xs_train = [X_1_train, ..., X_g_train]  # g is the number of parties
    ````

See [examples](../data/src).



