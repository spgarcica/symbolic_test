import optuna
from optuna.trial import TrialState
import numpy as np
from pysr import PySRRegressor


class OptunaBase:
    def __init__(self, name, num_trials=100):
        self.name = name
        self.num_trials = num_trials

    def create_or_load(self):
        storage = f"sqlite:///pysr_{self.name}.db"
        study_name = f"pysr_{self.name}"
        study = optuna.create_study(
            direction="minimize",
            storage=storage,
            study_name=study_name,
            load_if_exists=True,
        )
        return study

    def run(self, objective):
        study = self.create_or_load()
        study.optimize(objective, n_trials=self.num_trials)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        # for key, value in trial.params.items():
        #     print("    {}: {}".format(key, value))
        
        return trial.number



def objective(trial, X, y, X_val, y_val):
    # weights = [ trial.suggest_categorical(f"w_{i}", [1.0, 10.0]) for i in range(len(y)) ]
    weights = [ trial.suggest_float(f"w_{i}", 0.1, 2.0, step=0.1) for i in range(len(y)) ]

    model = PySRRegressor(
        niterations=100,  # < Increase me for better results
        binary_operators=["+", "-", "*", "/"],
        unary_operators=[
            "cos",
            "sin",
            "exp",
            "square",
            "sqrt",
            "inv(x) = 1/x",
            # ^ Custom operator (julia syntax)
        ],
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        # ^ Define operator for SymPy as well
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        # ^ Custom loss function (julia syntax)
        nested_constraints={
            "cos": {"cos": 0, "exp":0, "sin": 0},
            "sin": {"cos": 0, "exp":0, "sin": 0},
            "exp": {"cos": 0, "exp":0, "sin": 0},
        },
        verbosity=0,
    )

    model.fit(X, y, np.array([weights]))
    # model.fit(X, y)

    # get mse from predicting on validation set
    return ((model.predict(X_val) - y_val)**2).sum()



if __name__ == "__main__":
    import os, argparse
    from functools import partial
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--data_path", type=str, default="test")
    parser.add_argument("--num_trials", type=int, default=100)
    args = parser.parse_args()
    args.name = args.name[:-4] # to get rid of the .txt extension
    
    def get_data(data_path):
        with open(data_path, 'r') as f:
            data = f.read().splitlines()
        X = np.array([
            list(map(
                lambda e: float(e), x.split()[:-1]
            )) for x in data
        ])
        y = np.array([
            float(x.split()[-1]) for x in data
        ])
        return X, y
    
    root_dir = '../datasets/srsd-feynman_medium/train/'
    data_path = os.path.join(root_dir, args.data_path)
    X_train, y_train = get_data(data_path)
    X_train, X_val = X_train[:50], X_train[50:100]
    y_train, y_val = y_train[:50], y_train[50:100]
    
    optuna_base = OptunaBase(args.name, args.num_trials)
    best_trial = optuna_base.run(partial(
        objective,
        X=X_train,
        y=y_train,
        X_val=X_val,
        y_val=y_val
    ))
    print("Optuna run complete!")
    
    ## Clean up
    from glob import glob
    os.rename(sorted(glob('hall_of_fame*.csv'), key=os.path.getmtime)[best_trial], f'{args.name}.csv')
    for filename in glob(os.path.join('.', 'hall_of_fame*')):
        os.remove(filename)
    
    